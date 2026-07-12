from pathlib import Path
import re
import yaml

# ---- Beállítások ----
USERS_DIR = Path(r"Input/Users_v2")

BESS_BLOCK = {
    "bess_size": "12.0",
    "eta_bess_in": "0.98",
    "eta_bess_out": "0.96",
    "eta_bess_stor": "0.995",
    "eta_self_discharge": "1.0",
    "soc_bess_max": "0.9",
    "soc_bess_min": "0.1",
    "t_bess_min": "2",
}


def get_last_three_digits_from_filename(filename: str) -> str:
    """
    Kiszedi a fájlnévből az utolsó 3 számjegyet.
    Pl.: load_0420144653458813.yaml -> 813
    """
    stem = Path(filename).stem
    digits = re.findall(r"\d", stem)
    if len(digits) >= 3:
        return "".join(digits[-3:])
    return "N/A"


def insert_bess_after_pv(units_dict: dict) -> dict:
    """
    Új dict-et ad vissza, amiben a bess közvetlenül a pv után kerül be,
    ugyanarra a szintre, mint a pv, ue, ut, hss stb.
    """
    new_units = {}

    for key, value in units_dict.items():
        new_units[key] = value

        if key == "pv" and "bess" not in units_dict:
            new_units["bess"] = BESS_BLOCK.copy()

    return new_units


def main():
    yaml_files = sorted(list(USERS_DIR.glob("*.yaml")) + list(USERS_DIR.glob("*.yml")))

    if not yaml_files:
        print(f"Nincs YAML fájl itt: {USERS_DIR.resolve()}")
        return

    count_ue = 0
    count_ut = 0
    count_hss = 0
    count_pv = 0
    count_bess_added = 0
    count_bess_already_present = 0

    pv_file_suffixes = []

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            print(f"SKIP (nem dict struktúra): {yaml_file.name}")
            continue

        units = data.get("units", data)

        if not isinstance(units, dict):
            print(f"SKIP (nincs értelmezhető units blokk): {yaml_file.name}")
            continue

        has_ue = "ue" in units
        has_ut = "ut" in units
        has_hss = "hss" in units
        has_pv = "pv" in units
        has_bess = "bess" in units

        if has_ue:
            count_ue += 1
        if has_ut:
            count_ut += 1
        if has_hss:
            count_hss += 1
        if has_pv:
            count_pv += 1
            last3 = get_last_three_digits_from_filename(yaml_file.name)
            pv_file_suffixes.append((yaml_file.name, last3))

        modified = False

        if has_pv:
            if has_bess:
                count_bess_already_present += 1
                print(f"MÁR VAN BESS: {yaml_file.name} | utolsó 3 szám: {get_last_three_digits_from_filename(yaml_file.name)}")
            else:
                new_units = insert_bess_after_pv(units)

                if "units" in data:
                    data["units"] = new_units
                else:
                    data = new_units

                modified = True
                count_bess_added += 1
                print(f"BESS BESZÚRVA: {yaml_file.name} | utolsó 3 szám: {get_last_three_digits_from_filename(yaml_file.name)}")
        else:
            print(f"NINCS PV: {yaml_file.name}")

        if modified:
            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    data,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False
                )

    print("\n--- ÖSSZESÍTÉS ---")
    print(f"YAML fájlok száma: {len(yaml_files)}")
    print(f"ue szekciók száma: {count_ue}")
    print(f"ut szekciók száma: {count_ut}")
    print(f"hss szekciók száma: {count_hss}")
    print(f"pv szekciók száma: {count_pv}")
    print(f"új bess blokkok beszúrva: {count_bess_added}")
    print(f"már meglévő bess blokkok: {count_bess_already_present}")

    print("\nPV-s fájlok és az utolsó 3 számjegyük:")
    for fname, last3 in pv_file_suffixes:
        print(f"  {fname} -> {last3}")


if __name__ == "__main__":
    main()