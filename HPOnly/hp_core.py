"""PV + HP + BESS esetek. kW, kWh, °C.
A díjak kutatási modellparaméterek, nem aktuális díjszabás-állítások.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.linalg import expm


# A modellezett HP-k közös, állandó hatásfokai.
# A YAML esetleges COP/EER mezőit szándékosan nem használjuk.
COP_HEATING = 3.8
EER_COOLING = 3.8
HP_HEATING_TH_MAX_KW = 12.0
HP_COOLING_TH_MAX_KW = 12.0  # simulate_ata.py: p_th_max_kW = 12.0 mindkét irányban


@dataclass
class Household:
    key: str
    pv: np.ndarray
    base: np.ndarray
    fixed_hp: np.ndarray
    hp: dict | None = None
    outdoor: np.ndarray | None = None
    solar: np.ndarray | None = None
    mode: np.ndarray | None = None
    pf: np.ndarray | None = None
    hp_tariff: str = "GEO"
    bess: dict | None = None


@dataclass
class Settings:
    dt: float = .25
    low: float = 36.386
    high: float = 70.104
    annual_cap: float = 2523.
    geo: float = 23.368
    export: float = 5.25
    shared_low: float = 6.668  # energiadíj; RHD külön hozzáadva
    shared_high: float = 23.527
    shared_rhd: float = 29.718
    gap: float = .01
    time_limit: float = 1800.
    solver_log: bool = True
    penalty: float = 1_000_000.
    use_legacy_pf: bool = False  # True: korábbi implicit kiegészítő fűtés
    include_bess: bool = True
    morning: tuple = (6., 9.)
    evening: tuple = (17., 22.)


def vector(values, n, label, nonnegative=False):
    x = np.asarray(values, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f'{label}: hibás hossz/alak vagy nem véges adat')
    if nonnegative and np.any(x < -1e-9):
        raise ValueError(f'{label}: negatív teljesítmény')
    return np.maximum(x, 0) if nonnegative else x


def load_households(input_dir, sim_yaml, profiles_csv, year=2021, dt=.25,
                    max_users=105, only_user=None, pv_ratio=1., use_legacy_pf=False,
                    *, users_dir=None, hp_profiles_csv=None, solar_gain_csv=None,
                    weather_csv=None):
    """CSV villamos profilok: kWh/lépés; solar gain: kW; hőmérséklet: °C.
    A base kizárólag units.ue; units.ut, hss nem modellezett. BESS: units.bess.
    HP nélküli háztartás megmarad a közösségben. PV nélküli HP: rögzített GEO-profil.
    """
    root = Path(input_dir)
    users_root=Path(users_dir) if users_dir is not None else root/'users_v4'
    hp_path=Path(hp_profiles_csv) if hp_profiles_csv is not None else root/'heat_pump_profiles.csv'
    solar_path=Path(solar_gain_csv) if solar_gain_csv is not None else root/'solar_gain_profiles.csv'
    weather_path=Path(weather_csv) if weather_csv is not None else root/'homerseklet_sugarzas_interpolated_polynomial_fixed.csv'
    index = pd.date_range(f'{year}-01-01', f'{year+1}-01-01',
                          freq=pd.Timedelta(hours=dt), inclusive='left')
    n = len(index)
    if not np.isclose(24/dt, round(24/dt)):
        raise ValueError('dt maradék nélkül ossza a napot')
    sim = yaml.safe_load(Path(sim_yaml).read_text(encoding='utf-8')) or {}
    cache = {}
    def read_frame(path):
        path=Path(path)
        if path not in cache:
            frame=pd.read_csv(path)
            if len(frame)!=n:
                raise ValueError(f'{path}: {len(frame)} sor, várt {n}')
            timecol=next((c for c in ('timestamp','Unnamed: 0') if c in frame),None)
            if timecol is not None:
                times=pd.DatetimeIndex(pd.to_datetime(frame[timecol],errors='raise'))
                if not times.equals(index):
                    raise ValueError(f'{path}: az időbélyegek nem egyeznek a {year}-es {dt} órás időráccsal')
            else:
                raise ValueError(f'{path}: timestamp oszlop szükséges (Unnamed: 0 is elfogadott)')
            cache[path]=frame
        return cache[path]
    profiles = read_frame(profiles_csv)
    def series(file, col, energy=False):
        path = Path(file)
        frame = read_frame(path)
        if col not in frame: raise KeyError(f'{path}: hiányzó oszlop: {col}')
        return vector(frame[col], n, f'{file}:{col}', energy) / (dt if energy else 1)
    def electrical(col):
        if col in (None, ''): return np.zeros(n)
        if str(col) not in profiles: raise KeyError(f'Measurements profil hiányzik: {col}')
        return vector(profiles[str(col)], n, str(col), True) / dt
    households = []
    for rawkey in sim.get('users_list', []):
        key = str(rawkey)
        if key.lower() in {'battery','bess','community','load_battery'}: continue
        candidates = [users_root/key, users_root/f'{key}.yaml',users_root/f'{key}.yml']
        path = next((p for p in candidates if p.is_file()), None)
        if path is None: raise FileNotFoundError(f'Hiányzó user YAML: {key}')
        cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        units = cfg.get('units',cfg) or {}
        name = str((units.get('name') or {}).get('name',key))
        if only_user and only_user not in (key,name): continue
        pvblock=units.get('pv') or {}
        baseblock=units.get('ue') or {}
        pv = electrical(pvblock.get('profile_id') or pvblock.get('profile')) * pv_ratio
        base = electrical(baseblock.get('profile_id') or baseblock.get('profile'))
        block = units.get('heat_pump')
        fixed = np.zeros(n)
        hp = outdoor = solar = mode = pf = None
        if block:
            profile = str(block.get('profile_id') or block.get('profile') or '')
            if not profile: raise ValueError(f'{key}: HP profile_id hiányzik')
            fixed = series(hp_path,profile,True)
            if pv.sum() > 1e-9:
                hp = block
                weather = read_frame(weather_path)
                col = next((k for k in ('T','temperature','air_temperature','temp','T_out') if k in weather),None)
                if col is None: raise KeyError(f'Hőmérséklet oszlop hiányzik: {list(weather)}')
                outdoor = vector(weather[col],n,'T_out')
                solar = series(solar_path,profile)
                e = block.get('heat_pump_parameters') or {}
                mode = np.where(np.isin(index.month,[1,2,3,4,10,11,12]),1,
                       np.where(np.isin(index.month,[6,7,8]),-1,
                       np.where(outdoor < float(e.get('heating_balance_c',16)),1,
                       np.where(outdoor > float(e.get('cooling_balance_c',24)),-1,0))))
                # Korábbi részvételi faktor: nem villamos kiegészítő hőforrás.
                daysteps = int(round(24/dt))
                daily = outdoor.reshape(-1,daysteps).mean(axis=1)
                pf = np.repeat(np.select([daily<0,daily<=5,daily<=7],[0,.4,.7],default=1.),daysteps)
        if pf is not None and not use_legacy_pf:
            pf=np.ones(n)  # kizárólag a HP állítja elő a fűtési hőt
        battery=units.get('bess')
        if battery and pv.sum()<=1e-9:
            print(f'[BESS kihagyva] {key}: nincs PV-termelés')
            battery=None
        household=Household(key,pv,base,fixed,hp,outdoor,solar,mode,pf,
                            "A" if block and pv.sum()>1e-9 else "GEO",battery)
        if battery is not None: bess_parameters(household,dt)
        if hp is not None:
            _,_,parameters=thermal_parameters(household,dt)
            print(f'[HP] {key}: fűtési hőteljesítmény={parameters["ph"]:.3f} kW; '
                  f'hűtési hőteljesítmény={parameters["pc"]:.3f} kW; '
                  f'COP/EER={parameters["cop"]:g}/{parameters["eer"]:g}'
                  + '; közös kapacitásbeállítás')
        households.append(household)
        if only_user or len(households) >= max_users: break
    if not households: raise ValueError('Nincs kiválasztott háztartás')
    return households, index


def building_5r2c(h):
    """A saját YAML épületadataiból számol; nem ír vissza a YAML-be.
    A konstansok a feltöltött simulate_ata.py értékei.
    """
    raw = h.hp.get('household_parameters') or {}
    c = raw.get('model_5r2c')
    if c is not None:
        if not isinstance(c, dict):
            raise ValueError(f'{h.key}: model_5r2c nem szótár')
        return c
    fields = ('Aref', 'h', 'Htr', 'Hve', 'U_window', 'window_total')
    missing = [key for key in fields if key not in raw]
    if missing:
        raise ValueError(f'{h.key}: nincs model_5r2c; a household_parameters blokkból '
                         f'hiányzik: {", ".join(missing)}. A CSV sorai azonosító nélkül nem párosíthatók.')
    try:
        area, height, htr, hve, uw, aw = [float(raw[key]) for key in fields]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{h.key}: az épület hat alapadata szám legyen') from exc
    if not np.isfinite([area, height, htr, hve, uw, aw]).all() or min(area, height, htr, hve, uw, aw) <= 0:
        raise ValueError(f'{h.key}: az épület alapadatai pozitív, véges számok legyenek')
    # W/K: ablak, opak fal, tömeg-felület és belső felület-levegő.
    hw = uw * aw
    hop = htr - hw
    hms = 9.10 * 3.0 * area
    his = 3.45 * 4.5 * area
    if not 0 < hop < hms:
        raise ValueError(f'{h.key}: 0 < Htr - U_window*window_total < H_ms szükséges; '
                         f'Htr_op={hop:g}, H_ms={hms:g} W/K')
    hem = 1.0 / (1.0 / hop - 1.0 / hms)
    return {
        'C_air_kWh_per_K': 5.0 * 0.34 * area * height / 1000.0,
        'C_mass_kWh_per_K': 280.0 / 3600.0 * area,
        'H_em_kW_per_K': hem / 1000.0,
        'H_w_kW_per_K': hw / 1000.0,
        'H_ms_kW_per_K': hms / 1000.0,
        'H_is_kW_per_K': his / 1000.0,
        'H_v_kW_per_K': hve / 1000.0,
    }


def hp_capacities(h, c):
    """A simulate_ata.py közös, mindkét irányban 12 kW-os hőteljesítmény-korlátja.
    A globális beállítás elsőbbséget élvez a YAML kapacitásmezőivel szemben.
    Az épületadatok a hődinamikát határozzák meg, a HP méretét itt nem.
    """
    ph, pc = float(HP_HEATING_TH_MAX_KW), float(HP_COOLING_TH_MAX_KW)
    if not np.isfinite([ph, pc]).all() or min(ph, pc) <= 0:
        raise ValueError('HP_HEATING_TH_MAX_KW és HP_COOLING_TH_MAX_KW pozitív, véges szám legyen')
    return ph, pc, False, False


def thermal_parameters(h, dt):
    c = building_5r2c(h)
    e = h.hp.get('heat_pump_parameters') or {}
    required = ['C_air_kWh_per_K','C_mass_kWh_per_K','H_em_kW_per_K',
                'H_w_kW_per_K','H_ms_kW_per_K','H_is_kW_per_K','H_v_kW_per_K']
    missing = [key for key in required if key not in c]
    if missing:
        raise ValueError(f'{h.key}: model_5r2c hiányzó mezők: {", ".join(missing)}')
    try:
        v = [float(c[k]) for k in required]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{h.key}: az 5R2C paraméterei számok legyenek') from exc
    if not np.isfinite(v).all() or min(v)<=0:
        raise ValueError(f'{h.key}: az 5R2C paraméterei pozitív, véges számok legyenek')
    ca,cm,hem,hw,hms,his,hv = v
    den = hw+hms+his
    # Állapot x=[T_air,T_mass]. Egzakt diszkretizálás: nem instabil Euler.
    a = np.array([[ (his*(his/den-1)-hv)/ca, his*hms/den/ca],
                  [ hms*his/den/cm, (hms*(hms/den-1)-hem)/cm]])
    b = np.array([[(his*hw/den+hv)/ca,1/ca],[(hms*hw/den+hem)/cm,0]])
    aug = np.zeros((4,4)); aug[:2,:2]=a; aug[:2,2:]=b
    z = expm(aug*dt)
    cop, eer = float(COP_HEATING), float(EER_COOLING)
    if not np.isfinite([cop,eer]).all() or min(cop,eer)<=0:
        raise ValueError('COP_HEATING és EER_COOLING pozitív, véges szám legyen')
    ph,pc,ph_estimated,pc_estimated=hp_capacities(h,c)
    p = dict(cop=cop,eer=eer,ph=ph,pc=pc,ph_estimated=ph_estimated,pc_estimated=pc_estimated,
             heat=float(e.get('heating_setpoint_c',22)),cool=float(e.get('cooling_setpoint_c',25)),
             hcritical=float(e.get('heating_critical_c',20)),hemergency=float(e.get('heating_emergency_c',18)),
             ccritical=float(e.get('cooling_critical_c',26)),cemergency=float(e.get('cooling_emergency_c',27)),
             delta=float(e.get('precondition_delta_c',1)))
    if not (p['hemergency']<=p['hcritical']<=p['heat']<=p['cool']<=p['ccritical']<=p['cemergency']) or p['delta']<0:
        raise ValueError(f'{h.key}: hibás komfortküszöbök')
    return z[:2,:2],z[:2,2:],p


def blocked_at(step,s):
    hour=(step*s.dt)%24
    return any(a<=hour<b if a<=b else hour>=a or hour<b for a,b in [s.morning,s.evening])


def rule_based(h,s):
    n=len(h.pv)
    if h.hp is None:
        return add_rule_bess(h,{'hp_el':h.fixed_hp.copy(),'geo':h.fixed_hp.copy()},s)
    a,b,p=thermal_parameters(h,s.dt)
    states=np.zeros((n+1,2)); states[0]=p['heat']
    heat=np.zeros(n);cool=np.zeros(n);target=np.zeros(n);short=np.zeros(n)
    bp=bess_parameters(h,s.dt) if s.include_bess and h.bess is not None else None
    be=np.zeros(n+1);bc=np.zeros(n);bd=np.zeros(n);bg=np.zeros(n);bl=np.zeros(n)
    if bp: be[0]=bp['initial']
    for t in range(n):
        free=a@states[t]+b@np.array([h.outdoor[t],h.solar[t]])
        surplus=max(h.pv[t]-h.base[t],0)
        available=(min(bp['power'],max(bp['retention']*be[t]-bp['emin'],0)*bp['eta_out']/s.dt)
                   if bp else 0.)
        # A helyi tároló először az alapfogyasztás hiányát fedezi.
        local_supply=surplus+max(available-max(h.base[t]-h.pv[t],0),0)
        mode=h.mode[t]; blocked=blocked_at(t,s) and h.hp_tariff=="A"
        if mode==1:
            target[t]=p['heat']+(p['delta'] if surplus>1e-4 else 0)
            demand=max((target[t]-free[0])/b[0,1],0)
            if blocked:
                # PV csak kritikus, hálózat csak vészhelyzeti hőmérsékletnél.
                limit=(local_supply*p['cop']/h.pf[t] if h.pf[t]>0 else p['ph']) if states[t,0]<=p['hcritical'] else 0
                if states[t,0]<=p['hemergency']: limit=p['ph']
            else: limit=p['ph']
            heat[t]=min(demand,limit,p['ph'])
        elif mode==-1:
            target[t]=p['cool']-(p['delta'] if surplus>1e-4 else 0)
            demand=max((free[0]-target[t])/b[0,1],0)
            if blocked:
                limit=local_supply*p['eer'] if states[t,0]>=p['ccritical'] else 0
                if states[t,0]>=p['cemergency']: limit=p['pc']
            else: limit=p['pc']
            cool[t]=min(demand,limit,p['pc'])
        else: target[t]=p['heat']
        states[t+1]=free+b[:,1]*(heat[t]-cool[t])
        # Komforteltérés mindkét irányban: a passzív túlmelegedést is jelezzük.
        if mode==1: short[t]=max((p['hemergency'] if blocked else target[t])-states[t+1,0],
                                states[t+1,0]-(p['heat']+p['delta']),0)
        if mode==-1: short[t]=max(states[t+1,0]-(p['cemergency'] if blocked else target[t]),
                                 p['cool']-p['delta']-states[t+1,0],0)
        if bp:
            demand=h.base[t]+(h.pf[t]*heat[t]/p['cop']+cool[t]/p['eer'] if h.hp_tariff=='A' else 0)
            bc[t],bd[t],bg[t],be[t+1],bl[t]=battery_step(
                bp,be[t],max(h.pv[t]-demand,0),max(demand-h.pv[t],0),s.dt)
    electric=h.pf*heat/p['cop']+cool/p['eer']
    return dict(hp_el=electric,geo=electric.copy() if h.hp_tariff=='GEO' else np.zeros(n),
                air=states[:,0],mass=states[:,1],heat=heat,cool=cool,target=target,
                comfort_shortfall=short,aux_heat=(1-h.pf)*heat,
                bess_charge=bc,bess_discharge=bd,bess_grid_charge=bg,
                bess_energy=be,bess_soc=be/bp['size'] if bp else be.copy(),bess_loss=bl)


def settle(pv,demand,community):
    """Helyi önfogyasztás először, majd maradékigény-arányos megosztás."""
    own=np.minimum(pv,demand); deficit=np.maximum(demand-pv,0); surplus=np.maximum(pv-demand,0)
    shared=np.zeros_like(deficit); sold=np.zeros_like(surplus)
    if community:
        d=deficit.sum(axis=0);u=surplus.sum(axis=0);total=np.minimum(d,u)
        shared=deficit*np.divide(total,d,out=np.zeros_like(d),where=d>1e-12)
        sold=surplus*np.divide(total,u,out=np.zeros_like(u),where=u>1e-12)
    return dict(own=own,shared=shared,sold=sold,grid=deficit-shared,export=surplus-sold,
                deficit=deficit,surplus=surplus)


def bills(flows,geo,s,used=None):
    count=flows['grid'].shape[0]
    used=np.zeros(count) if used is None else np.asarray(used)
    # A keret az A-mérő összes vételezésére vonatkozik: hálózat + megosztás.
    grid=flows['grid'].sum(axis=1)*s.dt;share=flows['shared'].sum(axis=1)*s.dt
    total=grid+share; low=np.minimum(total,np.maximum(s.annual_cap-used,0));high=total-low
    # A havi elszámolás egységes átlagárat ad hálózati és megosztott energiára.
    fraction=np.divide(low,total,out=np.zeros_like(low),where=total>1e-12)
    cost=grid*(fraction*s.low+(1-fraction)*s.high)
    cost+=share*(fraction*s.shared_low+(1-fraction)*s.shared_high+s.shared_rhd)
    # Közösségi belső eladás átadás: a vevőnél költség, az eladónál bevétel.
    cost+=(geo.sum(axis=1)*s.geo-flows['export'].sum(axis=1)*s.export
           -flows['sold'].sum(axis=1)*s.export)*s.dt
    return cost,used+total


def bess_parameters(h, dt):
    """A YAML szerinti energiatároló. A tárolási tényezők időlépésenkéntiek."""
    c = h.bess
    if not isinstance(c, dict):
        raise ValueError(f'{h.key}: a bess blokk nem szótár')
    p = dict(size=float(c.get('bess_size', 0)),
             eta_in=float(c.get('eta_bess_in', .98)),
             eta_out=float(c.get('eta_bess_out', .96)),
             eta_stor=float(c.get('eta_bess_stor', .995)),
             eta_self=float(c.get('eta_self_discharge', 1.)),
             soc_min=float(c.get('soc_bess_min', .1)),
             soc_max=float(c.get('soc_bess_max', .9)),
             soc_init=float(c.get('soc_bess_init', c.get('soc_bess_min', .1))),
             hours=float(c.get('t_bess_min', 2)))
    if not np.isfinite(list(p.values())).all() or dt <= 0:
        raise ValueError(f'{h.key}: nem véges BESS-paraméter')
    if p['size'] <= 0 or p['hours'] <= 0:
        raise ValueError(f'{h.key}: BESS-kapacitás és t_bess_min pozitív legyen')
    if not 0 <= p['soc_min'] <= p['soc_init'] <= p['soc_max'] <= 1:
        raise ValueError(f'{h.key}: hibás SOC-határok/kezdeti SOC')
    if not all(0 < p[name] <= 1 for name in ('eta_in', 'eta_out', 'eta_stor', 'eta_self')):
        raise ValueError(f'{h.key}: a BESS hatásfokai (0,1] között legyenek')
    p.update(emin=p['size']*p['soc_min'], emax=p['size']*p['soc_max'],
             initial=p['size']*p['soc_init'], power=p['size']/p['hours'],
             retention=p['eta_stor']*p['eta_self'])
    if p['emin']*(1-p['retention']) > p['power']*p['eta_in']*dt + 1e-10:
        raise ValueError(f'{h.key}: a minimális SOC vesztesége nem pótolható a teljesítménykorláttal')
    return p


def battery_step(p, energy, pv_surplus, deficit, dt):
    """Csak saját PV-ből tölt; külső töltés kizárólag a SOC-minimum pótlására."""
    decayed = p['retention']*energy
    charge = min(max(pv_surplus, 0), p['power'],
                 max((p['emax']-decayed)/(p['eta_in']*dt), 0))
    discharge = 0.
    if charge <= 1e-10:
        discharge = min(max(deficit, 0), p['power'],
                        max(decayed-p['emin'], 0)*p['eta_out']/dt)
    before_repair = decayed + dt*(p['eta_in']*charge-discharge/p['eta_out'])
    repair = max(p['emin']-before_repair, 0)/(dt*p['eta_in'])
    next_energy = before_repair + dt*p['eta_in']*repair
    return charge, discharge, repair, next_energy, energy-decayed


def add_rule_bess(h, result, s):
    """Éves szabályalapú tároló; az állapot folyamatos a hónaphatárokon is."""
    n = len(h.pv)
    result.update(bess_charge=np.zeros(n), bess_discharge=np.zeros(n),
                  bess_grid_charge=np.zeros(n), bess_loss=np.zeros(n),
                  bess_energy=np.zeros(n+1), bess_soc=np.zeros(n+1))
    if not s.include_bess or h.bess is None:
        return result
    p = bess_parameters(h, s.dt)
    result['bess_energy'][0] = p['initial']
    demand = h.base + (result['hp_el'] if h.hp_tariff == 'A' else 0)
    for t in range(n):
        c, d, g, energy, loss = battery_step(p, result['bess_energy'][t],
            max(h.pv[t]-demand[t], 0), max(demand[t]-h.pv[t], 0), s.dt)
        for name, value in [('bess_charge',c), ('bess_discharge',d),
                            ('bess_grid_charge',g), ('bess_loss',loss)]:
            result[name][t] = value
        result['bess_energy'][t+1] = energy
    result['bess_soc'] = result['bess_energy']/p['size']
    return result


def settle_results(households, results, start, stop, community):
    """BESS utáni maradék megosztása; a tároló nem exportál és nem tölt REC-ből
    a SOC-minimum veszteségpótlásán felül. A GEO külön mérőn marad.
    Az own közvetlen PV + a helyben felhasznált akkumulátorkisütés.
    """
    pv = np.stack([h.pv[start:stop] for h in households])
    demand = np.stack([h.base[start:stop] + (r['hp_el'] if h.hp_tariff == 'A' else 0)
                       for h, r in zip(households, results)])
    zeros = np.zeros(stop-start)
    charge = np.stack([r.get('bess_charge', zeros) for r in results])
    discharge = np.stack([r.get('bess_discharge', zeros) for r in results])
    repair = np.stack([r.get('bess_grid_charge', zeros) for r in results])
    deficit = np.maximum(demand-pv, 0)-discharge+repair
    surplus = np.maximum(pv-demand, 0)-charge
    if np.min(deficit) < -1e-5 or np.min(surplus) < -1e-5:
        raise RuntimeError('A BESS helyi fogyasztáson/PV-feleslegen túli energiát használt')
    # Kerekítési hibákat a nemnegatív maradékokra korlátozzuk.
    flows = settle(np.maximum(surplus,0), np.maximum(deficit,0), community)
    flows['own'] = np.minimum(pv,demand)+discharge
    flows['pv_direct'] = np.minimum(pv,demand)
    return flows
