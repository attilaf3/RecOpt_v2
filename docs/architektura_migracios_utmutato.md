# EnergyCommunityFlexible – architektúra és migrációs útmutató

Állapotfelvétel: 2026. szeptember 11. Cél: közös megbeszélés a hallgatóval, majd az ő fejlesztéseinek beillesztése. Ez a dokumentum nem végez merge-et és nem módosít működési kódot.

Az „elkészült” állítások a jelenlegi munkakönyvtár kódjára vonatkoznak, nem feltétlenül a Gitben már commitolt állapotra. A korábbi felépítés leírása a beszélgetésben rögzített változtatások összegzése, nem teljes commitonkénti változásnapló. A „javaslat” és „nyitott kérdés” bekezdések még nem megvalósított funkciókat jelölnek.

## 1. Vezetői összefoglaló

A legfontosabb változás: a szcenárió már nem egy önállóan lemásolt program, hanem közös inputokra, modellekre és elszámolási komponensekre épülő futtatási konfiguráció. Ez még nem teljesen egységes rendszer: a BESS-modell és a BESS-eredménycsomag közös, de a hőszivattyús flexibilitás, a közösségi termikus optimalizáció és több vezérlés hiányzik.

A migráció során nem a hallgató fájljait kell egy az egyben visszamásolni a régi helyükre. A bennük lévő új fizikát, vezérlési szabályt, adatbeolvasást és ábrázolást kell külön azonosítani, majd a megfelelő közös rétegbe beilleszteni.

Három alapelv:

1. A fizikai üzem és a pénzügyi elszámolás külön fogalom. Ugyanaz a berendezésmenetrend többféleképpen elszámolható.
2. A név nem bizonyítja a megvalósítottságot. A `community` fájlnév nem feltétlenül közösségi eszközvezérlést jelent; a `HP optimizer` nem feltétlenül időben áthelyezhető fűtést jelent.
3. A mértékegység és az időindex az interfész része. A kWh/időlépés nem kW, és a T eleji SOC nem azonos a T végi SOC-val.

## 2. Régi és új felépítés

| Korábbi helyzet | Jelenlegi megoldás | Integrációs következmény |
|---|---|---|
| Külön bojler-, BESS- és kombinált egyéni optimalizáció, ismétlődő modellezéssel | Közös `optimize_household`, eszközkapcsolókkal | Az új eszközlogikát a közös modellbe kell bekötni |
| Több BESS-dinamika, eltérő végállapottal és hálózati töltéssel | Közös optimalizációs BESS-korlátok; közös greedy energiahatárok | Régi nyitott SOC és korlátlan hálózati töltés nem hozható vissza észrevétlenül |
| Szcenárióazonosítók és indítások szétszórva | Központi regiszter, futtatási terv, manifest | Új funkcióhoz a regiszter státuszát is ellenőrizni kell |
| Árak és elérési utak több helyen | Közös Config példány, INI-szekciók | Új futtató ne vezessen be saját abszolút adatútvonalat |
| Inputok többszöri feldolgozása | `SimulationInputs` központi adatmodell | Új olvasót az InputReading csomagban kell elhelyezni |
| Több közösségi számlaszámítás | Közös Economics mag, kompatibilitási adapterek | Az eltérő tarifakeret-politikákat továbbra is tudatosan kell kiválasztani |
| Greedy SOC végi, optimalizált SOC eleji indexeléssel | Nyilvánosan egységes start/end/boundary mezők | A régi greedy eredményfeldolgozás migrációt igényel |

### Felelősségi térkép

```text
run_scenario.py
  └─ Scenarios/scenario_manager.py
      ├─ nem optimalizált egyéni/közösségi futtató
      ├─ közös egyéni optimalizációs futtató
      └─ közösségi optimalizációs futtató

Futtatók közös építőelemei:
  Config → InputReading → fizikai modell/vezérlés
                             ├─ Optimization / eszközkorlátok
                             └─ Utility / greedy és elosztás
                      → Economics → eredménymentés → Visualization
```

Ez felelősségi ábra, nem szigorú importgráf: néhány régi modul még egyszerre végez futtatást, összesítést és mentést.

## 3. Melyik modul mire való?

| Forrás | Felelősség |
|---|---|
| [scenario_manager.py](C:/work/EnergyCommunityFlexible/Scenarios/scenario_manager.py) | Regiszter, kódnormalizálás, futtatási terv, részleges szcenáriók védelme, manifest |
| [configuration.py](C:/work/EnergyCommunityFlexible/Utility/configuration.py) | `Config`, közös `config` példány, típusos lekérdezés, útvonal-feloldás |
| [user_input_reading.py](C:/work/EnergyCommunityFlexible/InputReading/user_input_reading.py) | YAML/CSV inputok és `SimulationInputs` |
| [individual_optimizer.py](C:/work/EnergyCommunityFlexible/OptimizedIndividualScenarios/individual_optimizer.py) | Egy háztartás közös bojler–BESS optimalizációja |
| [run_individual_optimization.py](C:/work/EnergyCommunityFlexible/OptimizedScenarios/run_individual_optimization.py) | Háztartások iterálása, közös modell hívása, elszámolás és mentés |
| [optimize_disaggregated.py](C:/work/EnergyCommunityFlexible/OptimizedCommunityScenarios/optimize_disaggregated.py) | Közösségi optimalizáció, háztartásonkénti energiaáramok |
| [runtime.py](C:/work/EnergyCommunityFlexible/Optimization/runtime.py) | Solver-választás, fallback, státuszellenőrzés, numerikus kiolvasás |
| [bess_constraints.py](C:/work/EnergyCommunityFlexible/Optimization/bess_constraints.py) | Közös PuLP BESS-változók és korlátok |
| [boiler_constraints.py](C:/work/EnergyCommunityFlexible/OptimizedIndividualScenarios/boiler_constraints.py) | Egyéni bojler/HSS hőmérleg és üzemidőkorlátok |
| [optimization_constraints.py](C:/work/EnergyCommunityFlexible/OptimizedIndividualScenarios/optimization_constraints.py) | Kéttömbös tarifa és import/export kizárás közös segédei |
| [nonopt_common.py](C:/work/EnergyCommunityFlexible/NotOptimizedScenarios/nonopt_common.py) | Egyéni alap/greedy szimuláció, HP-profil bekötése, összesítés |
| [noopt_community_1b.py](C:/work/EnergyCommunityFlexible/NotOptimizedScenarios/noopt_community_1b.py) | Saját PV/BESS után megmaradó közösségi többlet elosztása |
| [bess_dispatch.py](C:/work/EnergyCommunityFlexible/Utility/bess_dispatch.py) | Greedy töltés/kisütés és minimum-SOC-pótlás energiahatárai |
| [bess_periodic.py](C:/work/EnergyCommunityFlexible/Utility/bess_periodic.py) | Greedy periodikus SOC-kezdőállapot keresése |
| [energy_allocation.py](C:/work/EnergyCommunityFlexible/Utility/energy_allocation.py) | Kapacitáskorlátos arányos/egyenlő kiosztás és kereslet–többlet párosítás |
| [calculate_economics.py](C:/work/EnergyCommunityFlexible/Economics/calculate_economics.py) | Tarifák, közösségi pénzáramok, számlaszámítás |
| [settlement_modes.py](C:/work/EnergyCommunityFlexible/Economics/settlement_modes.py) | `3d-K` és `4d-I` utólagos elszámolási átalakításai |
| [result_schema.py](C:/work/EnergyCommunityFlexible/Utility/result_schema.py) | Közös BESS-eredményséma és CSV-csomag |
| [profile_sampling.py](C:/work/EnergyCommunityFlexible/HeatPump/profile_sampling.py) | HP-paraméterek és setpointprofil mintavételezésének helye |
| [nonopt_plots.py](C:/work/EnergyCommunityFlexible/Visualization/nonopt_plots.py) | Kiszervezett nem optimalizált ábrázolások |

Az `individual_opt_boiler.py`, `individual_opt_bess.py`, `individual_opt_bess_boiler.py` kompatibilitási belépési pontok. Az `opt_boiler.py` és `opt_bess.py` szintén adapterek; ne ezekben induljon új fizikai modell. A régi egyéni BESS-korlátmodul a közös modult exportálja újra. A `noopt_individual_1a.py` törlésre került, régi importját meg kell szüntetni.

HP-kivétel: az [opt_hp.py](C:/work/EnergyCommunityFlexible/OptimizedIndividualScenarios/opt_hp.py) mögött létező modell rögzített HP-profilt szolgál ki PV/hálózat segítségével. Nem épületállapot-alapú flexibilis HP-optimalizáló. Ráadásul az [individual_opt_hp.py](C:/work/EnergyCommunityFlexible/OptimizedIndividualScenarios/individual_opt_hp.py) `p_*` nevű bemenetei dokumentáltan kWh/időlépés egységűek: ez örökölt interfészkivétel, nem követendő konvenció.

## 4. A szcenáriók tényleges állapota

| Kód | Regiszter státusza | Tényleges tartalom / korlát |
|---|---|---|
| `0-I` | implemented | Mért bojler és alapfogyasztás, BESS nélkül; HP termosztátos szimulációból is származhat, tehát nem minden adat feltétlenül mért |
| `0-K` | partial | Alapüzem és közösségi megosztás; a közösségi ágban nincs HP-modell |
| `1d-I` | partial | Greedy BESS; teljes időprogramos bojler/HP rule-based vezérlés még nincs |
| `1d-K` | partial | Lokális PV/BESS után közösségi megosztás; HP és teljes rule-based eszközvezérlés hiányzik |
| `2d-K` | unavailable | Nincs közösségi rule-based eszközkoordinátor |
| `3d-I` | partial | Közös egyéni bojler–BESS optimalizáció; flexibilis HP nélkül |
| `3d-K` | partial | Ugyanez, utólagos közösségi elszámolással |
| `4d-I` | partial | Közösségi optimalizáció után egyéni elszámolás; BESS optimalizált, bojlerprofil rögzített, HP hiányzik |
| `4d-K-E` | partial | Közösségi energetikai cél, ugyanilyen eszközkorlátokkal |
| `4d-K-C` | partial | Közösségi pénzügyi cél, ugyanilyen eszközkorlátokkal |
| `5d-I`, `5d-K` | unavailable | Nincs DSO/aggregátori célfüggvény és hálózati korlátmodell |

Fontos pontosítás: a regiszter szövege a közösségi optimalizációnál „BESS és bojler” támogatást említ. A kód alapján ez nem közösségi bojler-HSS optimalizáció: a `p_el_heater` adott idősor, a modell annak energiaellátását osztja fel.

Részleges modell csak `allow_partial=True` vagy `--allow-partial` engedéllyel indul. Ez nem kapcsolja be a hiányzó funkciót. A regiszter érzékenységvizsgálati címkéi sem jelentenek automatikusan kész paramétersöprést.

Ellenőrizhető belépési pontok a projekt gyökeréből:

```powershell
python run_scenario.py --list
python run_scenario.py --describe 3d-K
python run_scenario.py 3d-K --allow-partial --max-users 2 --mip --out results/migration-smoke
```

Az utolsó parancs valódi futtatás: helyes inputok és solver szükségesek. A `--max-users` a háztartásszámot korlátozza, nem az időhorizontot. Rövid, szintetikus modellellenőrzéshez közvetlen modellhívás kell.

## 5. Inputok és konfiguráció

A `SimulationInputs` idősorai jellemzően `(T, U)` alakúak: időlépés × háztartás. Az eszközparaméterek U elemű vektorok, a `user_names` sorrendje kötelezően az oszlopsorrend.

- Villamos `e_*_kwh`: kWh/időlépés.
- Villamos `p_*_kw`: kW, azaz `e / dt`.
- DHW: liter/időlépésből számított hőenergia és hőteljesítmény; nem közvetlenül villamos fogyasztás.
- BESS SOC-paraméterek: kapacitásarányok; a modellállapot kWh.
- `eta_bess_stor`: időlépésenkénti megtartási tényező. Időfelbontás-váltásnál nem másolható át automatikusan változatlanul.

Új adatforrás bekötési mintája, a jelenlegi API szerint:

```python
from Utility.configuration import config
from InputReading.user_input_reading import read_simulation_inputs

inputs = read_simulation_inputs(
    config.getpath("paths", "simulation_yaml"),
    config.getpath("paths", "profiles_csv"),
    config.getpath("paths", "dhw_profiles_csv"),
    max_users=2,
    dt=config.getfloat("simulation", "dt_hours"),
    n_steps=96,
)
```

A Config relatív útvonalakat a projekt gyökeréhez old fel, nem az INI könyvtárához. Alapesetben a `config/config.ini` fájlt olvassa; az `ENERGY_COMMUNITY_CONFIG` környezeti változóval másik INI adható meg, még a modulok importálása előtt.

A fontos szekciók: `paths`, `tariffs`, `simulation`, `bess`, `optimization`, `scenario`, `scenario_outputs`, `visualization`. A közösen verziózandó minta: [example_config.ini](C:/work/EnergyCommunityFlexible/config/example_config.ini). Az aktív `config.ini` Git-ignore alatt van.

Nyitott technikai kérdés: több alapértelmezett függvényargumentum és tarifakonstans importáláskor olvassa ki a konfigurációt. A `config.set()` későbbi hívását ezek nem feltétlenül követik. Érzékenységvizsgálatnál adjunk explicit paramétert/Tariffs objektumot, vagy induljon új folyamat az új konfigurációval. Továbbá fizikai állandók és vezérlési alapértékek még maradtak a kódban; nem igaz, hogy már minden szám az INI-ből jön.

## 6. Közös optimalizációs komponensek

Az `optimize_household` egy háztartás egy dimenziós kW-idősorait fogadja. A `include_boiler` és `include_bess` kapcsolókkal választ eszközöket. A `include_heat_pump=True` jelenleg explicit hibát ad, nem csendben közelít.

Kis szintetikus migrációs példa:

```python
from OptimizedIndividualScenarios.individual_optimizer import optimize_household

result = optimize_household(
    p_pv=[2.0, 0.0], p_ue=[0.5, 0.5], p_dhw=[0.0, 0.0],
    dt=0.25, include_boiler=False, include_bess=True,
    size_bess=1.0, eta_bess_stor=1.0,
    soc_bess_min=0.0, soc_bess_init=0.5, soc_bess_max=1.0,
    bess_min_mode_steps=1, run_lp=False, msg=False,
    allow_grid_charge_for_min_soc=False,
)
assert abs(result["e_bess_boundary"][0] - result["e_bess_boundary"][-1]) < 1e-6
```

A solver-kezelést ne másoljuk új modellbe: a `solve_problem` közös solver-választást és státuszellenőrzést ad. A visszaadott numerikus értékeket csak elfogadott megoldási státusz után használjuk; a kiolvasásban használt alapérték önmagában nem bizonyítja a megoldás érvényességét.

Új eszközkorlátok ajánlott mintája: külön builder, explicit bemenetek, visszaadott változók, a hívóban felépített közös villamos mérleg. A `build_bess` már prefixet használ a több háztartásból álló modell névütközéseinek elkerülésére. A bojlersegédek jelenlegi korlátnevei nem háztartásprefixesek: közösségi HSS-bővítés előtt ezt is rendezni kell.

## 7. BESS: mi változott fizikailag?

A közös optimalizációs állapotegyenlet:

```text
E[t+1] = retention * E[t]
         + dt * eta_in * (P_pv_charge[t] + P_grid_charge[t])
         - dt * P_discharge[t] / eta_out
E[T] = E[0]
```

A teljesítménykorlát a kapacitás és a minimális töltési/kisütési idő hányadosa. A kezdeti SOC az optimalizációban rögzített; a régi `open` végfeltétel és az `unrestricted` hálózati töltés elutasított.

MIP-ben két bináris engedélyezi a töltést és a kisütést. A megengedett állapotok `00`, `10`, `01`; a `11` tiltott. A bináris 1 engedélyt jelent, nem feltétlenül pozitív teljesítményt. A minimum üzemmódhossz az időszak végén–elején is érvényes. LP-ben ezek az üzemmódkorlátok nincsenek érvényesítve; LP-eredményből nem szabad automatikusan megvalósítható kapcsolási menetrendet állítani.

A `allow_grid_charge_for_min_soc` alapértelmezése false. Bekapcsolva a modell pontosan a minimumállapothoz hiányzó energiát engedi hálózatból pótolni. Ez a pontos maximumfüggvény LP-kérés mellett is bináris guardot igényelhet. Önkisülés és kevés PV mellett a ciklikusság hálózati pótlás nélkül infeasible lehet: ezt nem szabad utólagos mesterséges töltéssel eltüntetni.

| Energiaút | Jelenlegi közös optimalizációs bekötés |
|---|---|
| Saját PV → saját BESS | Engedett |
| Saját BESS → saját fogyasztás | Engedett |
| Saját BESS → saját bojler | A tarifás bekötésnél |
| Hálózat → BESS | Csak opcionális minimum-SOC-pótlás |
| Közösségi megosztás → BESS | Nincs bekötve |
| BESS → közösségi értékesítés/export | Nincs bekötve |

Greedy oldalon a közös segédek energiahatárokat számolnak, nem egyetlen teljes közös vezérlőállapot-gépet. A sorrend: önkisülés, PV közvetlenül a jogosult terhelésekre, PV-többlet a BESS-be / BESS a hiányra, majd opcionális minimum-SOC-pótlás a fennmaradó töltési kapacitásból. Az egyéni ág üzemmódzárát megtartottuk; a közösségi greedy ágban nincs azonos zár.

A periodikus greedy inicializálás legfeljebb 128 iterációban keresi azt a kezdő SOC-t, amelyre ugyanoda jut vissza a teljes időszak végén. Ez változtathatja a kezdeti 50%-os tippet, és jelentősen növelheti a futásidőt. A zárak teljes belső állapotának és a tárolt energia eredetének periodikusságát nem szabad ebből automatikusan következtetni. A lokális/hálózati energiaeredet és az SSI kezelése külön ellenőrzendő, főleg engedélyezett grid top-up mellett.

## 8. Bojler, HSS és HP: mit ne keverjünk össze?

A közös egyéni bojlerkorlátok hőmérlegből, DHW-kiszolgálásból, hőveszteségből, teljesítmény- és üzemidőkorlátokból állnak. Az időszak vége a kezdő hőmérsékletre kapcsolódik. A napon belüli működési korlátok megléte optimalizálóban nem jelent kész, oksági rule-based bojlert.

Közösségi HSS megvalósításához szükséges: háztartásonkénti termikus állapot, DHW, komfortkorlát, fűtési döntés, megfelelő villamos mérleg és névprefixek. A rögzített mért bojlerprofil ellátásának átrendezése nem helyettesíti ezt.

Flexibilis HP-hoz szükséges: épület termikus állapota, komfort, időfüggő környezeti hatások, HP-határteljesítmény és COP, valamint a vezérlés információkészlete. A már létező termosztátos profilgenerálás és a fix profil ellátási optimalizációja felhasználható, de nem nevezhető teljes HP-flexibilitásnak.

## 9. Fizikai üzem és elszámolás különválasztása

`3d-K`: először háztartásonkénti optimalizáció történik. Ezután az időlépésenkénti jogosult A-tarifás import és export párosul; a berendezések menetrendje nem módosul. B/GEO importot ez az adapter nem von be a virtuális megosztásba.

`4d-I`: a közösségileg optimalizált menetrend után a belső vásárlás a megfelelő mérő hálózati importja, a belső értékesítés hálózati export lesz az elszámolásban. Nincs új egyéni optimalizáció.

A közös számlázási mag ellenére két tarifakeret-politika maradt: `proportional` és `grid_first`. Példa: 1 kWh megmaradt kedvezményes keret, ugyanabban a lépésben 1 kWh hálózati és 1 kWh közösségi vásárlás. Arányosan 0,5–0,5 kWh jut kedvezményes keretbe; grid-first esetben 1–0 kWh. Ez elszámolási döntés, nem refaktorálási részlet.

A greedy közösségi kiosztás vevőoldalon arányos vagy egyenlő kvótás, eladóoldalon arányos a többlettel. A virtuális adapter a kiválasztott módot mindkét oldalon használja. Az `equal` kapacitáskorlátos vízfeltöltés: az alacsony igényű szereplő maradék kvótáját újraosztja.

Nyitott jogosultsági kérdés: egyes közösségi ágak a B tarifás bojlerhez is rendelhetnek megosztott energiát, miközben a virtuális `3d-K` adapter csak A importot von be. Ezt kutatási feltevésként egyeztetni kell; a kód nem bizonyítja a valós mérési/elszámolási megengedettséget.

Az `1d-I` és `1d-K` jelenleg külön szimulátorágakon fut. Eltérhet a HP-lefedettség, az üzemmódzár és a komponensbontás. Ezért a két jelenlegi futtatásra még nem állítható, hogy kizárólag az elszámolás különbözik. A kívánt kontrollált összehasonlításhoz egy közös fizikai futás két elszámolása javasolt.

## 10. Eredmények: kötelező migrációs szabályok

Minden fő BESS-futtató azonos `bess/` csomagot ment. A `schema.json` 1-es verziót, `dt_hours`, `steps`, `users`, indexelést és mezőnkénti mértékegységet tartalmaz.

| Mező | Hossz | Jelentés |
|---|---|---|
| `e_bess_boundary` | T+1 | Teljes állapotpálya, kezdet és vég is |
| `e_bess_start` | T | `boundary[:-1]` |
| `e_bess_end` | T | `boundary[1:]` |
| `e_bess` | T | Nyilvános eredményben a start aliasa; a közös csomagban az explicit start mező használatos |
| `e_pv_to_bess`, `e_grid_to_bess` | T | AC oldali kWh/időlépés |
| `e_bess_to_load`, `e_bess_out` | T | AC oldali kisütött kWh/időlépés |
| `e_bess_in` | T | PV- és hálózati töltés összege |
| `p_pv_bess`, `p_grid_bess`, `p_bess_in`, `p_bess_out` | T | Átlagos kW az időlépésben |

A CSV-k oszlopai a háztartások, nincs külön időbélyegoszlop. Az időbélyeg, naptár, időzóna és nyári időszámítás egységes kezelése további feladat. A pénzügyi és termikus kimenetek továbbra is szcenárióspecifikusak: nem készült teljes univerzális eredményobjektum.

Migrációs példa:

```python
# Régi greedy kód: e_bess[-1] az időszak végi állapot volt.
# Új nyilvános eredmény:
end_energy = result["timeseries"]["e_bess_boundary"][-1]
per_step_end = result["timeseries"]["e_bess_end"]
```

A belső `_simulate_*` egyutas greedy függvények még végi SOC-t adnak a periodikus inicializálónak. Új külső kód a nyilvános `simulate_*` belépési pontot használja. T+1 állapotot ne tegyünk ugyanabba a DataFrame-be T hosszú energiaáramokkal.

## 11. A megbeszélt információszegény rule-based vizsgálat

Ez a beszélgetésben tisztázott kutatási irány, még nem kész implementáció:

- BESS látja a helyi PV/fogyasztás egyenlegét, de nem lát más háztartást.
- Bojler előre rögzített nappali időablakot és setpointprofilt követ, termelési visszacsatolás nélkül.
- Ugyanez a szabály PV nélküli bojleren is vizsgálható. A nappali napsütés általános előzetes tudás, nem valós idejű közösségi információ.
- A közösségi elszámolás utólag párosíthatja az egyidejű többletet és fogyasztást; nem kell hozzá, hogy a bojler közösségi jelzést kapjon.

Javasolt összehasonlítás: alapüzem / csak PV-s bojlerek időprogramja / minden bojler időprogramja; mindegyikhez egyéni és közösségi elszámolás ugyanazon fizikai menetrenden. A komfort, DHW, időjárás, eszközállomány és véletlen mag azonos legyen. A nappali setpointot ne a tesztév teljes jövőbeli PV-görbéjéből válasszuk ki.

Mérendő: számla, import/export, csúcsterhelés, hőveszteség, komfortsértés, saját és közösségi SSI/SCI külön definícióval. Azonos fizikai működés mellett pusztán az elszámolás nem javítja a fizikai közösségi önellátást. A háztartási allokált mutató ettől változhat, ezt külön kell megnevezni.

A `0` mért bojlerprofil és az új termikus szimuláció közötti összevetéshez a termikus alapüzemet előbb kalibrálni kell. Különben a különbség egy része modellhiba, nem vezérlési előny.

## 12. A hallgató kódjának beillesztési menete

1. **Állapot rögzítése.** A saját és a hallgatói munkát külön azonosítható Git-állapotban őrizzük meg. A jelenlegi munkafa sok módosított és új fájlt tartalmaz; ne legyen reset vagy teljes könyvtár-felülírás.
2. **Funkcionális leltár.** Minden hallgatói függvényhez írjuk oda: új fizika, vezérlés, adatolvasás, elszámolás, megjelenítés vagy régi infrastruktúra másolata.
3. **Interfészleltár.** Bemeneti alak, egység, időindex, tarifa, információigény, kezdő/végállapot, kimenet. Külön ellenőrizzük a HP örökölt egységkivételét.
4. **Kis referenciaeset.** Mentsünk ki néhány lépéses inputot és elvárt mérlegeket a hallgatói megoldásból. Előbb különbséget diagnosztizáljunk, ne automatikusan a régi számot tekintsük igaznak.
5. **Adapter, majd integráció.** Elsőként csak a mértékegységet és bemeneti szerkezetet alakítsuk át. Utána kerüljön a szabály a megfelelő modellbe; az inputbeolvasás és solver-kód ne költözzön vele.
6. **Egy funkció egy változtatás.** Külön fejlesztés legyen a nappali bojlervezérlés, HP-dinamika, közösségi koordinátor és kimeneti átalakítás.
7. **Közös korlátok használata.** BESS-builder, tarifasegédek és solver-runtime maradjanak közösek. A megváltozó fizikát külön dokumentáljuk és teszteljük.
8. **Kettős elszámolás.** Ahol a kutatási kérdés ezt igényli, egyszer számítsuk a berendezések fizikai menetrendjét, kétszer az elszámolást.
9. **Regiszter és dokumentáció.** Csak a tényleges funkciók és tesztek megléte után javuljon a szcenárió státusza. Ne töröljünk korlátozást pusztán azért, mert fut a program.
10. **Review és merge.** A két fél együtt nézze át a modellváltozást, az interfészt és az eredményeket; a merge külön jóváhagyott lépés legyen.

Git-csapda: a `.gitignore` általános `run_*` szabályt tartalmaz. Új futtató esetén ellenőrizni kell, hogy ténylegesen verziózott lesz-e; néhány közös futtatónak már van kivétele, a gyökérbeli indítót is külön ellenőrizni kell. Az INI-minta és a működéshez szükséges új modulok bekerülése ugyanolyan fontos, mint a modellkódé.

## 13. Tesztelési és átvételi terv

A jelenlegi tesztek a `tests` könyvtárban vannak. Helyi futtatás, ha a korábban telepített tesztfüggőségek rendelkezésre állnak:

```powershell
$env:PYTHONPATH='.codex-test-deps'
python -m unittest discover -s tests
```

A `.codex-test-deps` helyi, nem verziózott segédkönyvtár; a hallgató környezetében külön reprodukálható függőségtelepítés szükséges. A tesztkimenetben a kihagyott teszteket is ellenőrizni kell: hiányzó PuLP/PyYAML mellett egyes tesztek skip státuszt adnak. A korábbi 32 tesztes siker nem teljes éves validáció és nem bizonyít minden szcenáriót.

Átvételi minimum:

- [ ] Input: T/U alak, oszlopsorrend, hiányzó profilok, DHW konverzió, eltérő dt.
- [ ] BESS: minden időlépés mérlege, hatásfok és önkisülés, teljesítményhatár, SOC-határok és ciklikusság.
- [ ] BESS: tiltott/engedett minimum-pótlás, PV elsőbbsége, közös PV+grid töltési korlát, nem megoldható eset explicit hibája.
- [ ] MIP: töltés/kisütés kizárása és horizonton átnyúló minimum üzemmód; LP-eredmények külön címkézése.
- [ ] Bojler/HP: melegvíz/fűtési komfort, hőmérleg, hőveszteség, teljesítmény és vezérlési információkorlát.
- [ ] Megosztás: kapott összeg = átadott összeg időlépésenként; kiosztás nem nagyobb igénynél/többletnél; egy energia nem többször kiosztott.
- [ ] Elszámolás: A/B/GEO jogosultság, tarifaküszöb átlépése, proportional/grid-first, közösségi belső fizetések egyeztetése.
- [ ] Páros vizsgálat: I/K elszámolásváltás nem változtatja a rögzített fizikai menetrendet.
- [ ] Eredmény: T és T+1 helyes, kW/kWh megfelelő, metaadat és háztartási sorrend egyezik.
- [ ] Előbb 1 háztartás/néhány lépés, majd 2–3 háztartás/néhány nap, csak utána szezonális és éves futás.
- [ ] Solver-státusz, futásidő, memória, MIP-rés és kezdőállapot rögzítése; ugyanazon véletlen magok és bemenetek használata.

## 14. Mit kell a megbeszélés végére eldönteni?

1. Az `1d` egységesen információszegény helyi vezérlés legyen-e, a PV nélküli bojlerek opcionális nappali programjával? A legutóbbi megbeszélés ezt indokolja.
2. Mi a konkrét bojler időablak, setpoint, tartalékfelfűtés és komfortfeltétel? Fix kutatási szabály vagy külön tanítási időszakon választott paraméter?
3. Mely mérőkörök jogosultak közösségi energiára, és melyik tarifakeret-politika érvényes minden összehasonlításban?
4. A BESS energiaútjai maradnak-e lokálisak, vagy új változatban közösségi töltés/export is kell? Ez új fizikát és elszámolást igényel.
5. Az egyéni és közösségi greedy üzemmódzár különbségét megtartjuk vagy egységesítjük? Hogyan kezeljük a tárolt energia eredetét és periodikus inicializálását?
6. Melyik hallgatói fejlesztés kerüljön be először: nappali bojler, flexibilis HP, közösségi HSS vagy `2d-K` koordinátor?
7. Mik a pontos SSI/SCI-definíciók, különösen közösségi allokáció és hálózatból töltött BESS mellett?

Javasolt első integráció: közös termikus bojler-alapmodell és információszegény nappali vezérlés, kis referenciaesettel; erre ugyanazon fizikai menetrend egyéni/közösségi számlázása. Ez közvetlenül válaszol a kutatási kérdésre, és nem igényli előbb a teljes DSO- vagy HP-optimalizáció elkészítését.
