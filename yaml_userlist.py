from __future__ import annotations

from pathlib import Path
import re
import yaml


# =========================
# BEÁLLÍTÁSOK
# =========================

INPUT_YAML = Path("Input/simulation_config_disaggregated.yaml")
OUTPUT_YAML = Path("Input/simulation_config_disaggregated_with_userlist.yaml")

# Az a mappa, amelyben a user fájlok vannak
USERS_FOLDER = Path("Input/Users")

# Ha kell a hosszabb nevű YAML-hez hasonlóan az elejére:
ADD_LOAD_BATTERY = True


# =========================
# SEGÉDFÜGGVÉNYEK
# =========================

def extract_user_name(file_path: Path) -> str | None:
    """
    Egy fájlnévből kinyeri a user nevet.

    Példák:
        0420144653422463.csv        -> load_0420144653422463
        load_0420144653422463.csv   -> load_0420144653422463
        timeseries_0420144653422463.csv -> load_0420144653422463
    """

    stem = file_path.stem

    # Ha timeseries_ előtag van, levesszük
    if stem.startswith("timeseries_"):
        stem = stem.replace("timeseries_", "", 1)

    # Ha már load_ előtaggal van, megtartjuk
    if stem.startswith("load_"):
        return stem

    # Ha hosszú numerikus user ID van benne
    match = re.search(r"\d{10,}", stem)
    if match:
        return f"load_{match.group(0)}"

    return None


def collect_users_from_folder(folder: Path) -> list[str]:
    """
    Beolvassa a mappában lévő fájlokból a user neveket.
    """

    if not folder.exists():
        raise FileNotFoundError(f"Nem található a mappa: {folder}")

    users = []

    for file_path in folder.iterdir():
        if file_path.is_file():
            user = extract_user_name(file_path)
            if user is not None:
                users.append(user)

    # Duplikációk törlése, rendezés
    users = sorted(set(users))

    if ADD_LOAD_BATTERY:
        users = ["load_battery"] + users

    return users


# =========================
# FUTTATÁS
# =========================

def main() -> None:
    with open(INPUT_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    users_list = collect_users_from_folder(USERS_FOLDER)

    config["users_list"] = users_list

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"Kész: {OUTPUT_YAML}")
    print(f"Hozzáadott userek száma: {len(users_list)}")


if __name__ == "__main__":
    main()