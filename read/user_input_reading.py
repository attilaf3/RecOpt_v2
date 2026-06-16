import yaml


# TODO: befejezni, Lilla csinálja, csv readinget is
def read_users(sim_yaml_path, max_users=None, exclude=None):
    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    raw_users_list = sim.get("users_list")
    if raw_users_list:
        users_list = list(raw_users_list)
    else:
        users_dir = sim_yaml_path.parent / "Users"
        if not users_dir.exists():
            raise RuntimeError("A simulation YAML nem tartalmaz users_list-et, és a Users könyvtár sem található: "
                               f"{users_dir}")
        users_list = sorted(p.stem for p in users_dir.iterdir() if p.is_file() and p.suffix.lower() == ".yaml")
        if not users_list:
            raise RuntimeError(
                "A simulation YAML nem tartalmaz users_list-et, és a Users könyvtárban sincs .yaml fájl.")

    if max_users is not None:
        users_list = users_list[:int(max_users)]

    # exclude some pseudo users by name (optional)
    if exclude is None:
        exclude = {"battery", "bess", "community"}
    users_list = [u for u in users_list if str(u).strip().lower() not in exclude]
    if not users_list:
        raise RuntimeError("A simulation YAML nem tartalmaz users_list-et vagy max_users=0.")
    return users_list


def read_dhw():
    pass


def read_profiles():
    pass
