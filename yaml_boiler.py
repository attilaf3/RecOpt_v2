from pathlib import Path
import yaml

# ---- Beállítások ----
USERS_DIR = Path(r"Input/Users_v2")   # ezt módosítsd, ha más a mappa útvonala

# Csak a bojler / HSS hőmérsékletparaméterei
HSS_TEMPERATURE_PARAMS = {
    "T_env": "20.0",
    "T_in": "10.0",
    "T_max": "65.0",
    "T_min": "40.0",
    "T_out": "40.0",
    "T_set": "50.0",
}


def update_hss_temperature_params(units_dict: dict) -> bool:
    """
    Módosítja a hss blokk hőmérsékletparamétereit.
    True-t ad vissza, ha történt módosítás.
    """
    if "hss" not in units_dict:
        return False

    if not isinstance(units_dict["hss"], dict):
        return False

    modified = False

    for key, new_value in HSS_TEMPERATURE_PARAMS.items():
        old_value = units_dict["hss"].get(key)

        if old_value != new_value:
            units_dict["hss"][key] = new_value
            modified = True

    return modified


def main():
    yaml_files = sorted(list(USERS_DIR.glob("*.yaml")) + list(USERS_DIR.glob("*.yml")))

    if not yaml_files:
        print(f"Nincs YAML fájl itt: {USERS_DIR.resolve()}")
        return

    count_ue = 0
    count_ut = 0
    count_hss = 0
    count_pv = 0
    count_hss_modified = 0
    count_hss_missing = 0
    count_hss_unchanged = 0

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

        if has_ue:
            count_ue += 1
        if has_ut:
            count_ut += 1
        if has_pv:
            count_pv += 1

        if has_hss:
            count_hss += 1

            modified = update_hss_temperature_params(units)

            if modified:
                count_hss_modified += 1
                print(f"HSS HŐMÉRSÉKLETEK MÓDOSÍTVA: {yaml_file.name}")

                if "units" in data:
                    data["units"] = units
                else:
                    data = units

                with open(yaml_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        data,
                        f,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False
                    )
            else:
                count_hss_unchanged += 1
                print(f"HSS MÁR EZEKEN AZ ÉRTÉKEKEN VAN: {yaml_file.name}")

        else:
            count_hss_missing += 1
            print(f"NINCS HSS: {yaml_file.name}")

    print("\n--- ÖSSZESÍTÉS ---")
    print(f"YAML fájlok száma: {len(yaml_files)}")
    print(f"ue szekciók száma: {count_ue}")
    print(f"ut szekciók száma: {count_ut}")
    print(f"pv szekciók száma: {count_pv}")
    print(f"hss szekciók száma: {count_hss}")
    print(f"hss hőmérsékletek módosítva: {count_hss_modified}")
    print(f"hss már változatlan volt: {count_hss_unchanged}")
    print(f"hss nélküli fájlok: {count_hss_missing}")


if __name__ == "__main__":
    main()