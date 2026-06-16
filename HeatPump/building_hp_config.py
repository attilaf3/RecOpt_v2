from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BuildingHPParams:
    """
    Egy épület + hőszivattyú paraméterezése.

    Ezekből számolunk:
      - Q_HL(T_out): hőigény
      - T_su(T_out): fűtési görbe
      - ΔT(T_out)   : hőlépcső
      - COP(T_out)  : hőszivattyú hatásfoka
      - P_el(T_out) : szükséges villamos teljesítmény, ha Q_HP = Q_HL
      - b0, b1      : Q(T_out) = b0 + b1 * T_out együtthatók
    """
    name: str
    H_tot_W_per_K: float  # TABULA-ból (W/K)
    T_set: float = 23.0  # komfort / belső tervezési hőmérséklet [°C]

    # Fűtési görbéhez:
    T_out_design: float = -18.0  # tervezési külső [°C] (hideg pont)
    T_out_mild: float = 23.0  # enyhe külső [°C] (meleg pont)
    T_su_nom: float = 45.0  # névleges előremenő [°C] (hidegben)
    T_su_min: float = 30.0  # minimum előremenő [°C] (enyhébb időben)

    # Lineáris COP(ΔT) = a0 + a1 * ΔT (egyedi épületenként)
    cop_a0: float = 5.06
    cop_a1: float = -0.05
    cop_min: float = 0.5  # alsó korlát a COP-ra, numerikai védelem

    # Q(T_out) = b0 + b1 * T_out együtthatók (automatikusan számoljuk)
    b0: float | None = None
    b1: float | None = None

    def __post_init__(self):
        """
        Q_HL(T_out) = H_tot*(T_set - T_out)
                    = -H_tot * T_out + H_tot*T_set
        → b1 = -H_tot, b0 = H_tot*T_set
        Ha a user nem ad meg b0,b1-et, akkor innen számoljuk.
        """
        if self.b0 is None or self.b1 is None:
            self.b1 = -self.H_tot_W_per_K
            self.b0 = self.H_tot_W_per_K * self.T_set

    # ---------- HŐIGÉNY ----------

    def q_hl(self, T_out: np.ndarray) -> np.ndarray:
        """
        Hőigény a külső hőmérséklet függvényében:
            Q_HL(T_out) = H_tot * (T_set - T_out)
        Ha T_out >= T_set, akkor Q_HL = 0 (nem fűtünk).
        """
        T_out = np.asarray(T_out, dtype=float)
        Q_HL = self.H_tot_W_per_K * (self.T_set - T_out)
        # fűtés nem kell, ha kint melegebb van, mint a T_set
        Q_HL = np.maximum(Q_HL, 0.0)
        return Q_HL  # [W]

    def q_hl_affine(self, T_out: np.ndarray) -> np.ndarray:
        """
        Ugyanez b0, b1 formában:
            Q(T_out) = b0 + b1 * T_out
        (elsősorban ellenőrzés / b0,b1 demonstráció miatt).
        """
        T_out = np.asarray(T_out, dtype=float)
        Q = self.b0 + self.b1 * T_out
        Q = np.maximum(Q, 0.0)
        return Q

    # ---------- FŰTÉSI GÖRBE ----------
    def heating_curve(self, T_out: np.ndarray) -> np.ndarray:
        """
        Fűtési görbe:
            T_su(T_out) lineárisan két pont között:
               (T_out_design, T_su_nom) és (T_out_mild, T_su_min),
        majd levágva [T_su_min, T_su_nom] tartományra.
        """
        T_out = np.asarray(T_out, dtype=float)

        # Meredekség a két pontból
        m_su = (self.T_su_nom - self.T_su_min) / (self.T_out_design - self.T_out_mild)

        # Nyers lineáris előremenő hőmérséklet
        T_su_raw = self.T_su_min + m_su * (T_out - self.T_out_mild)

        # Levágás a minimum/névleges tartományra
        T_su = np.clip(T_su_raw, self.T_su_min, self.T_su_nom)

        return T_su  # [°C]

    # ---------- ΔT és COP ----------

    def deltaT_and_cop(self, T_out: np.ndarray) -> Dict[str, np.ndarray]:
        """
        ΔT(T_out) és COP(T_out) kiszámítása:
            ΔT = T_su(T_out) - T_out
            COP = a0 + a1 * ΔT, levágva cop_min alá.
        """
        T_out = np.asarray(T_out, dtype=float)

        # 1) T_su(T_out) fűtési görbéből
        T_su = self.heating_curve(T_out)

        # 2) ΔT = T_sink - T_source = T_su - T_out (levegő–víz HP)
        deltaT = T_su - T_out

        # 3) COP(ΔT)
        COP = self.cop_a0 + self.cop_a1 * deltaT
        COP = np.maximum(COP, self.cop_min)

        return {
            "T_su": T_su,
            "deltaT": deltaT,
            "COP": COP,
        }

    # ---------- P_el, ha Q_HP = Q_HL ----------

    def pel_if_cover_hl(self, T_out: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Ha a hőszivattyú pontosan fedezi a hőigényt (Q_HP = Q_HL),
        akkor ebből mennyi villamos teljesítmény kell:

            P_el(T_out) = Q_HL(T_out) / COP(T_out)

        Visszaad:
            - Q_HL_W        : hőigény [W]
            - T_su_degC     : előremenő hőmérséklet [°C]
            - deltaT_degC   : hőlépcső [K]
            - COP           : hatásfok [-]
            - P_el_W        : villamos teljesítmény [W]
        """
        T_out = np.asarray(T_out, dtype=float)

        Q_HL_W = self.q_hl(T_out)
        dc = self.deltaT_and_cop(T_out)
        T_su = dc["T_su"]
        deltaT = dc["deltaT"]
        COP = dc["COP"]

        # ahol COP>0, ott számolunk P_el-t; máshol 0
        P_el_W = np.zeros_like(Q_HL_W)
        mask = COP > 0.0
        P_el_W[mask] = Q_HL_W[mask] / COP[mask]

        return {
            "Q_HL_W": Q_HL_W,
            "T_su_degC": T_su,
            "deltaT_degC": deltaT,
            "COP": COP,
            "P_el_W": P_el_W,
        }


# ===== Példa: 4 különböző épület =====
if __name__ == "__main__":
    # 4 épület – H_tot és COP paraméterek egyedileg
    buildings: List[BuildingHPParams] = [
        BuildingHPParams(
            name="Haz_A",
            H_tot_W_per_K=197.0,
            cop_a0=5.06,
            cop_a1=-0.05,
        ),
        BuildingHPParams(
            name="Haz_B",
            H_tot_W_per_K=112.0,
            cop_a0=5.06,
            cop_a1=-0.05,
        ),
        BuildingHPParams(
            name="Haz_C",
            H_tot_W_per_K=107.0,
            cop_a0=5.06,
            cop_a1=-0.05,
        ),
        BuildingHPParams(
            name="Haz_D",
            H_tot_W_per_K=110.0,
            cop_a0=5.06,
            cop_a1=-0.05,
        ),
    ]

    # Külső hőmérséklet tartomány a diagramokhoz
    T_out_range = np.linspace(-20.0, 25.0, 200)

    # ----- Kiírjuk b0, b1, Q_nom értékeket mind a 4 házra -----
    print("Épületek Q(T_out) = b0 + b1*T_out paraméterei és névleges hőigényei:")
    for b in buildings:
        # b0, b1 már __post_init__-ben kiszámolódott
        Q_nom_W = b.H_tot_W_per_K * (b.T_set - b.T_out_design)
        print(
            f"{b.name}: "
            f"H_tot = {b.H_tot_W_per_K:.1f} W/K, "
            f"b0 = {b.b0:.2f}, b1 = {b.b1:.2f}, "
            f"Q_nom = {Q_nom_W / 1000.0:.2f} kW (T_out_design = {b.T_out_design} °C)"
        )
    print()

    # ----- Mind a 4 házra kiszámoljuk az eredményeket -----
    results_per_building: Dict[str, Dict[str, np.ndarray]] = {}

    for b in buildings:
        res = b.pel_if_cover_hl(T_out_range)
        results_per_building[b.name] = res

    # ----- Plot 1: Q_HL(T_out) mind a 4 házra -----
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    for b in buildings:
        res = results_per_building[b.name]
        plt.plot(T_out_range, res["Q_HL_W"] / 1000.0, label=b.name)
    plt.axhline(0, linestyle="--", color="grey")
    plt.xlabel("T_out [°C]")
    plt.ylabel("Q_HL [kW]")
    plt.title("Épület hőigény Q_HL(T_out)")
    plt.grid(True)
    plt.legend()

    # ----- Plot 2: T_su(T_out) mind a 4 házra -----
    plt.subplot(1, 3, 2)
    for b in buildings:
        res = results_per_building[b.name]
        plt.plot(T_out_range, res["T_su_degC"], label=b.name)
    plt.xlabel("T_out [°C]")
    plt.ylabel("T_su [°C]")
    plt.title("Fűtési görbe T_su(T_out)")
    plt.grid(True)
    plt.legend()

    # ----- Plot 3: COP(T_out) mind a 4 házra -----
    plt.subplot(1, 3, 3)
    for b in buildings:
        res = results_per_building[b.name]
        plt.plot(T_out_range, res["COP"], label=b.name)
    plt.xlabel("T_out [°C]")
    plt.ylabel("COP [-]")
    plt.title("Hőszivattyú COP(T_out)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ----- Extra: P_el(T_out) mind a 4 házra (ha Q_HP = Q_HL) -----
    plt.figure(figsize=(7, 4))
    for b in buildings:
        res = results_per_building[b.name]
        plt.plot(T_out_range, res["P_el_W"] / 1000.0, label=b.name)
    plt.xlabel("T_out [°C]")
    plt.ylabel("P_el [kW]")
    plt.title("Szükséges P_el(T_out), ha Q_HP = Q_HL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ----- Extra: SFH-02 / Haz_A fűtési görbe és COP két subploton -----
    # Tegyük fel, hogy Haz_A felel meg az SFH-02 épületnek
    sfh02 = buildings[0]  # BuildingHPParams(name="Haz_A", ...)

    # Újraszámoljuk az SFH-02 eredményeit (vagy használhatnánk a dict-et is)
    sfh02_res = sfh02.pel_if_cover_hl(T_out_range)
    T_su_sfh02 = sfh02_res["T_su_degC"]
    COP_sfh02 = sfh02_res["COP"]

    plt.figure(figsize=(10, 4), dpi=300)

    # Bal: fűtési görbe T_su(T_out)
    plt.subplot(1, 2, 1)
    plt.plot(T_out_range, T_su_sfh02)
    plt.xlabel(r"$T_\mathrm{out}$ [°C]")
    plt.ylabel(r"$T_\mathrm{su}$ [°C]")
    plt.title("Fűtési görbe")
    plt.grid(True)

    # Jobb: COP(T_out)
    plt.subplot(1, 2, 2)
    plt.plot(T_out_range, COP_sfh02)
    plt.xlabel(r"$T_\mathrm{out}$ [°C]")
    plt.ylabel(r"$\mathrm{COP}$ [-]")
    plt.title(r"$\mathrm{COP}(T_\mathrm{out})$")
    plt.grid(True)

    plt.tight_layout()
    # plt.savefig("sfh02_heating_curve_cop.png", dpi=300)
    plt.show()

    # ----- Extra: SFH-02 / Haz_A fűtési görbe + COP egy diagramon, két y tengellyel -----
    # Tegyük fel, hogy Haz_A felel meg az SFH-02 épületnek
    sfh02 = buildings[0]  # BuildingHPParams(name="Haz_A", ...)

    # Újraszámoljuk az SFH-02 eredményeit
    sfh02_res = sfh02.pel_if_cover_hl(T_out_range)
    T_su_sfh02 = sfh02_res["T_su_degC"]
    COP_sfh02 = sfh02_res["COP"]

    plt.figure(figsize=(6, 4), dpi=300)

    ax1 = plt.gca()            # bal oldali y tengely (T_su)
    ax2 = ax1.twinx()          # jobb oldali y tengely (COP)

    # Bal y: fűtési görbe
    line1, = ax1.plot(
        T_out_range,
        T_su_sfh02,
        label=r"$T_\mathrm{su}$"
    )
    ax1.set_xlabel(r"$T_\mathrm{out}$ [°C]")
    ax1.set_ylabel(r"$T_\mathrm{su}$ [°C]")
    ax1.grid(True)

    # Jobb y: COP
    line2, = ax2.plot(
        T_out_range,
        COP_sfh02,
        linestyle="-",
        label=r"$\mathrm{COP}$",
        color="orange"
    )
    ax2.set_ylabel(r"$\mathrm{COP}$ [-]")

    # Cím
    ax1.set_title(r"Fűtési görbe és $\mathrm{COP}$")

    # Közös legenda (mindkét tengelyről)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()
    # opcionálisan mentés:
    plt.savefig("sfh02_heating_curve_cop_twinaxis.png", dpi=300)
    plt.show()
