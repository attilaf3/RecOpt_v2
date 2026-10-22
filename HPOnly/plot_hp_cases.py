"""HP + PV + BESS esetek összevetése. IDE: Shift+F10; a beállítások lent.
Csak eredményeket olvas. Nem igényli a régi compare_rec_cases.py modult.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# ======================== BEÁLLÍTÁSOK ========================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / 'results_hp_bess'
# BESS nélküli korábbi futásnál: SCRIPT_DIR / 'results_hp_only'
OUTPUT_DIR = SCRIPT_DIR / 'figures_hp_bess'
CASES = ('0-I', '0-K', '1d-I', '1d-K', '3-HP-I', '4-HP-K-C_proportional')
BASELINE_CASE = '0-I'
HOUSEHOLD = 'load_0420144653433567'  # csak a szám is megadható; None: első modellezett HP
WINDOW_DAYS = 3
SEASONS = (('Tél', '01-15'), ('Tavasz', '04-15'), ('Nyár', '07-15'), ('Ősz', '10-15'))
DETAILED_CASES = ('1d-I', '3-HP-I', '4-HP-K-C_proportional')
BATTERY_CASE = '3-HP-I'  # ha hiányzik, más elkészült esetben keres aktív BESS-t
BATTERY_DAY = None  # pl. '2021-06-15'; None: tipikus aktív nap
DPI = 200
SAVE_PDF = False
MAKE_DETAILS = True
# ============================================================

COLORS = {'0-I': '#64748B', '0-K': '#94A3B8', '1d-I': '#3485A5',
          '1d-K': '#58A9BD', '3-HP-I': '#D18B40', '4-HP-K-C_proportional': '#369277'}
SHORT = {'4-HP-K-C_proportional': '4-HP-K-C'}
EPS = 1e-8


def safe(value):
    return re.sub(r'[^\w.-]+', '_', str(value))


def col(frame, name, optional=False):
    if name not in frame:
        if optional:
            return np.zeros(len(frame))
        raise ValueError(f'Hiányzó oszlop: {name}')
    values = pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f'Nem véges adatok: {name}')
    return values


def read_series(path):
    frame = pd.read_csv(path)
    if frame.empty or 'timestamp' not in frame:
        raise ValueError(f'Üres vagy timestamp nélküli fájl: {path}')
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp'), errors='raise'))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f'Ismétlődő vagy rendezetlen időbélyegek: {path}')
    frame.index = index
    return frame


def time_step(index):
    if len(index) < 2:
        raise ValueError('Legalább két időpont szükséges')
    # Az asi8 a dtype egységét használja (s/ms/us/ns), ezért nem szabad
    # automatikusan nanoszekundumként osztani. A Timedelta egységfüggetlen.
    differences = index[1:] - index[:-1]
    step = differences[0]
    if pd.isna(step) or step <= pd.Timedelta(0) or not np.all(differences == step):
        raise ValueError('Az idősor nem egyenletes időlépésű')
    return float(step / pd.Timedelta(hours=1))


def same_time_grid(first, second):
    # DatetimeIndex.equals eltérő tárolási felbontásnál (us/ns) akkor is
    # lehet False, amikor a tényleges időpontok azonosak.
    return len(first) == len(second) and bool(np.all(first == second))


def full_year(index):
    dt = time_step(index)
    year = index[0].year
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01',
                             freq=pd.Timedelta(hours=dt), inclusive='left',tz=index.tz)
    if not same_time_grid(index, expected):
        raise ValueError(
            f'Nem teljes naptári év: {len(index)} sor, '
            f'első={index[0]}, utolsó={index[-1]}, lépés={dt:g} óra; '
            f'várt: {len(expected)} sor, {expected[0]} - {expected[-1]}'
        )
    return dt


def save(fig, out, name, dpi):
    fig.savefig(out / f'{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    if SAVE_PDF:
        fig.savefig(out / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[ÁBRA] {out / (name + ".png")}')


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(alpha=.22, color='#94A3B8')
    ax.spines[['top', 'right']].set_visible(False)


def load_cases(root):
    loaded = {}
    rejected = []
    for label in CASES:
        path = root / label
        required = [path / 'summary.json', path / 'household_summary.csv',
                    path / 'community_timeseries.csv']
        if not all(p.is_file() for p in required):
            state = 'részleges/nem kész' if path.exists() else 'hiányzó'
            print(f'[KIHAGYVA] {label}: {state}; nem kerül az éves összevetésbe')
            continue
        summary = json.loads(required[0].read_text(encoding='utf-8-sig'))
        bills = pd.read_csv(required[1], dtype={'household': str})
        if not {'household', 'bill_ft'}.issubset(bills.columns):
            raise ValueError(f'{required[1]}: household és bill_ft szükséges')
        if bills.empty or bills['household'].duplicated().any():
            raise ValueError(f'{required[1]}: üres vagy ismétlődő háztartások')
        bills = bills.set_index('household')
        col(bills, 'bill_ft')
        try:
            series = read_series(required[2])
            dt = full_year(series.index)
        except ValueError as exc:
            message=f'{label}: {exc}; fájl: {required[2]}'
            rejected.append(message)
            print(f'[KIHAGYVA] {message}')
            continue
        pv = float(col(series, 'p_pv_kw').sum() * dt)
        demand = float((col(series, 'p_base_kw') + col(series, 'p_hp_el_kw')).sum() * dt)
        used = float((col(series, 'p_pv_self_kw') + col(series, 'p_shared_kw')).sum() * dt)
        if not np.isclose(float(summary['bill_ft']), bills['bill_ft'].sum(), rtol=1e-7, atol=.1):
            raise ValueError(f'{label}: a summary.json és a háztartási számlák eltérnek')
        if int(summary['households']) != len(bills):
            raise ValueError(f'{label}: eltérő háztartásszám az összesítőkben')
        if loaded:
            first = next(iter(loaded.values()))
            if set(bills.index) != set(first['bills'].index):
                raise ValueError(f'{label}: eltér a háztartáslista; azonos populációt hasonlíts össze')
            if not same_time_grid(series.index,first['series'].index):
                message=f'{label}: eltér az időrács/év az első elfogadott esethez képest'
                rejected.append(message)
                print(f'[KIHAGYVA] {message}')
                continue
        loaded[label] = dict(path=path, summary=summary, bills=bills, series=series, dt=dt,
                             SCI=used / pv if pv > EPS else np.nan,
                             SSI=used / demand if demand > EPS else np.nan)
    if not loaded:
        detail='\n'+'\n'.join(rejected) if rejected else ''
        raise ValueError(f'Nincs összevethető, teljes éves eset itt: {root}.{detail}')
    return loaded


def annual_plots(cases, out, dpi):
    rows = []
    for label, case in cases.items():
        frame = case['series']; dt = case['dt']; summary = case['summary']
        rows.append(dict(case=label, households=len(case['bills']), SCI=case['SCI'], SSI=case['SSI'],
                         bill_ft=float(case['bills']['bill_ft'].sum()),
                         mean_bill_ft=float(case['bills']['bill_ft'].mean()),
                         median_bill_ft=float(case['bills']['bill_ft'].median()),
                         grid_kwh=float(col(frame, 'p_grid_import_total_kw').sum()*dt),
                         shared_kwh=float(col(frame, 'p_shared_kw').sum()*dt),
                         export_kwh=float(col(frame, 'p_pv_export_kw').sum()*dt),
                         comfort_degree_hours=summary.get('comfort_shortfall_degree_hours', np.nan)))
    table = pd.DataFrame(rows).set_index('case')
    labels = list(cases); palette = [COLORS.get(label, '#64748B') for label in labels]
    names = [SHORT.get(label, label) for label in labels]
    if BASELINE_CASE in table.index:
        reference = float(table.loc[BASELINE_CASE, 'bill_ft'])
        table['saving_ft_vs_0I'] = reference - table['bill_ft']
        table['saving_pct_vs_0I'] = 100*(reference-table['bill_ft'])/reference if abs(reference)>EPS else np.nan
    table.to_csv(out / 'case_comparison_plot.csv', encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    positions=[]
    for label, color in zip(labels, palette):
        x, y = table.loc[label, ['SSI', 'SCI']]
        if np.isfinite([x, y]).all():
            ax.scatter(x, y, color=color, s=115, marker='s' if '-K' in label else 'o',
                       edgecolor='white', linewidth=1.2, label=SHORT.get(label, label), zorder=4)
            overlap=sum(abs(x-a)<.02 and abs(y-b)<.02 for a,b in positions)
            positions.append((x,y))
            ax.annotate(SHORT.get(label, label), (x,y), xytext=(6,6 if overlap%2==0 else -14),
                        textcoords='offset points', fontsize=8)
    ax.set(xlabel='SSI - önellátás', ylabel='SCI - önfogyasztás', title='HP + BESS: SCI-SSI összevetés')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, max(.1, float(table['SSI'].max())*1.18) if table['SSI'].notna().any() else 1)
    ax.set_ylim(0, max(.1, float(table['SCI'].max())*1.15) if table['SCI'].notna().any() else 1)
    style(ax); ax.legend(fontsize=8, loc='best'); save(fig,out,'sci_ssi',dpi)

    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    bars = ax.bar(names, table['bill_ft']/1e6, color=palette)
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set(ylabel='Összes éves számla [millió Ft]', title='Éves villamosenergia-költség')
    ax.margins(y=.18); style(ax); save(fig,out,'eves_osszkoltseg',dpi)

    paired = pd.DataFrame({label:case['bills']['bill_ft'] for label,case in cases.items()}).sort_index()
    paired.to_csv(out/'paired_household_bills.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(11,5), layout='constrained')
    bp = ax.boxplot([paired[label].to_numpy()/1000 for label in labels], patch_artist=True,
                    widths=.55, showmeans=True, meanprops={'marker':'D','markerfacecolor':'white',
                    'markeredgecolor':'#183A50','markersize':5}, medianprops={'color':'#183A50','linewidth':2})
    for patch,color in zip(bp['boxes'],palette): patch.set(facecolor=color,alpha=.72)
    ax.set_xticks(range(1,len(names)+1),names)
    ax.set(ylabel='Éves háztartási számla [ezer Ft]', title=f'Azonos {len(paired)} háztartás számlái; rombusz: átlag')
    style(ax); save(fig,out,'szamla_boxplot',dpi)

    if 'saving_ft_vs_0I' in table:
        fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
        bars=ax.bar(names,table['saving_ft_vs_0I']/1000,color=palette)
        annotations=[f'{value:.1f}%' if np.isfinite(value) else 'n.a.' for value in table['saving_pct_vs_0I']]
        ax.bar_label(bars, labels=annotations,padding=4)
        ax.axhline(0,color='#183A50',lw=.8); ax.margins(y=.2)
        ax.set(ylabel='Éves megtakarítás [ezer Ft]',title=f'Megtakarítás a {BASELINE_CASE} alapesethez képest')
        style(ax); save(fig,out,'megtakaritas_alapesethez',dpi)

    fig, axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    x=np.arange(len(labels)); width=.24
    for offset,(column,label,color) in enumerate([('grid_kwh','Hálózati import, GEO-val','#3182BD'),
                                                ('shared_kwh','Megosztott energia','#369277'),
                                                ('export_kwh','Hálózati export','#CB5555')]):
        axes[0].bar(x+(offset-1)*width,table[column]/1000,width,label=label,color=color)
    axes[0].set_xticks(x,names,rotation=20)
    axes[0].set(ylabel='Éves energia [MWh]',title='Külső és közösségi energiaforgalom')
    axes[0].legend(fontsize=8)
    axes[1].bar(names,table['comfort_degree_hours'],color=palette)
    axes[1].tick_params(axis='x',rotation=20)
    axes[1].set(ylabel='Összes komforteltérés [°C·h]',title='Modellezett HP-k komforteltérése')
    for ax in axes:style(ax)
    fig.suptitle('Alapeset: nincs épületmodell, ezért a komforteltérés nem értékelhető',fontsize=9)
    # A baseline 0-s diagnosztikája nem jelent igazolt komfortot.
    for i,label in enumerate(labels):
        if label.startswith('0-'):
            axes[1].text(i,0,'n.a.',ha='center',va='bottom',fontsize=8)
    save(fig,out,'energia_es_komfort',dpi)
    return table


def household_key(cases, choice):
    reference=next(iter(cases.values()))
    keys=list(reference['bills'].index)
    def canonical(value):
        value=Path(str(value)).stem
        return value.removeprefix('timeseries_').removeprefix('load_')
    if choice is not None:
        matches=[key for key in keys if canonical(key)==canonical(choice)]
        if len(matches)!=1:raise ValueError(f'Nem egyértelmű/hiányzó háztartás: {choice}')
        return matches[0]
    preferred=sorted(cases,key=lambda label:0 if label=='3-HP-I' else 1 if label=='1d-I' else 2)
    for label in preferred:
        for key in keys:
            path=cases[label]['path']/f'{key}_annual.csv'
            if path.is_file() and 'air' in pd.read_csv(path,nrows=0).columns:return key
    return keys[0]


def household_series(cases,key):
    loaded={}
    for label,case in cases.items():
        path=case['path']/f'{key}_annual.csv'
        if not path.is_file():
            print(f'[KIHAGYVA] Háztartási idősor hiányzik: {path}')
            continue
        frame=read_series(path)
        if not same_time_grid(frame.index,case['series'].index):raise ValueError(f'{path}: eltérő időrács')
        loaded[label]=frame
    if not loaded:raise FileNotFoundError(f'Nincs éves háztartási idősor: {key}')
    return loaded


def window(frame,date,days):
    start=pd.Timestamp(date); stop=start+pd.Timedelta(days=days)
    result=frame.loc[(frame.index>=start)&(frame.index<stop)]
    if result.empty or result.index[0]!=start or result.index[-1]+pd.Timedelta(hours=time_step(frame.index))!=stop:
        raise ValueError(f'Hiányos vagy nem illeszkedő ablak: {start} - {stop}')
    return result


def dates(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=9))


def blocked_shading(ax,index,settings):
    for day in pd.date_range(index[0].normalize(),index[-1].normalize(),freq='D'):
        for begin,end in [settings.get('morning',[6,9]),settings.get('evening',[17,22])]:
            left=day+pd.Timedelta(hours=begin);right=day+pd.Timedelta(hours=end)
            if end<=begin:right+=pd.Timedelta(days=1)
            ax.axvspan(left,right,color='#64748B',alpha=.08,zorder=0)


def seasonal_comparison(series,key,out,days,dpi):
    year=next(iter(series.values())).index[0].year
    for season,date in SEASONS:
        chunks={label:window(frame,f'{year}-{date}',days) for label,frame in series.items()}
        fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True,layout='constrained')
        for label,frame in chunks.items():
            color=COLORS.get(label,'#64748B');name=SHORT.get(label,label)
            axes[0].step(frame.index,col(frame,'p_hp_el_kw'),where='post',color=color,label=name,lw=1.1)
            if 'air' in frame:axes[1].plot(frame.index,col(frame,'air'),color=color,label=name,lw=1.3)
            if 'bess_soc' in frame and np.max(col(frame,'bess_energy_kwh',True))>EPS:
                axes[2].plot(frame.index,col(frame,'bess_soc'),color=color,label=name,lw=1.3)
        axes[0].set(ylabel='HP villamos teljesítmény [kW]',title=f'{key} | {season} | esetek összevetése')
        axes[1].set_ylabel('Belső levegő [°C]')
        axes[2].set(ylabel='BESS SOC',xlabel='Idő'); axes[2].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            style(ax)
            if ax.lines:ax.legend(ncol=3,fontsize=8)
            else:ax.text(.5,.5,'Ehhez az ábrához nincs modellezett állapot',ha='center',transform=ax.transAxes)
        dates(axes[-1]);save(fig,out,f'osszevetes_{safe(key)}_{safe(season)}',dpi)


def detailed_plot(frame,label,key,season,out,dpi,settings):
    fig,axes=plt.subplots(4,1,figsize=(16,13),sharex=True,layout='constrained',
                          gridspec_kw={'height_ratios':[2,1.2,1.2,1]})
    zeros=np.zeros(len(frame));x=frame.index
    charge=col(frame,'p_bess_charge_pv_kw',True)+col(frame,'p_bess_maintenance_charge_kw',True)
    sources=[('PV',col(frame,'p_pv_kw'),'#E7B34A'),
             ('BESS kisütés',col(frame,'p_bess_discharge_kw',True),'#8C6DB0'),
             ('Hálózat (A + GEO)' if np.max(np.abs(col(frame,'p_hp_geo_kw')))>EPS else 'Hálózat (A tarifa)',
              col(frame,'p_grid_import_total_kw'),'#3182BD'),
             ('REC-vételezés',col(frame,'p_shared_kw'),'#369277')]
    sinks=[('Alapfogyasztás',col(frame,'p_base_kw'),'#64748B'),
           ('HP',col(frame,'p_hp_el_kw'),'#E08145'),('BESS töltés',charge,'#B39ACB'),
           ('Hálózati export',col(frame,'p_pv_export_kw'),'#CB5555'),
           ('REC-eladás',col(frame,'p_shared_sold_kw'),'#70B69B')]
    residual=sum((v for _,v,_ in sources),zeros.copy())-sum((v for _,v,_ in sinks),zeros.copy())
    max_error=float(np.max(np.abs(residual)))
    if max_error>1e-4:raise ValueError(f'{label}/{key}: hibás villamos mérleg, max. eltérés={max_error:g} kW')
    for rows,sign in [(sources,1),(sinks,-1)]:
        active=[(name,values,color) for name,values,color in rows if np.max(np.abs(values))>EPS]
        if active:axes[0].stackplot(x,*[sign*v for _,v,_ in active],labels=[n for n,_,_ in active],
                                   colors=[c for _,_,c in active],step='post',alpha=.85)
    axes[0].axhline(0,color='#183A50',lw=.8)
    axes[0].set(ylabel='Teljesítmény [kW]',title=f'{SHORT.get(label,label)} | {key} | {season}: villamos mérleg')
    axes[0].legend(ncol=3,fontsize=8,loc='upper left')
    if 'air' in frame:
        axes[1].plot(x,col(frame,'air'),label='Belső levegő',color='#3182BD')
        axes[1].plot(x,col(frame,'mass'),label='Épülettömeg',color='#8C6DB0',lw=1)
        axes[1].plot(x,col(frame,'target'),label='Vezérlési cél/küszöb',color='#E08145',ls='--')
        axes[1].plot(x,col(frame,'T_out_C'),label='Külső hőmérséklet',color='#64748B',alpha=.65)
        axes[1].legend(ncol=4,fontsize=8)
    else:axes[1].text(.5,.5,'Rögzített HP-profil: nincs épületmodell',ha='center',transform=axes[1].transAxes)
    axes[1].set_ylabel('Hőmérséklet [°C]')
    axes[2].step(x,col(frame,'p_hp_el_kw'),label='HP villamos',where='post',color='#E08145')
    for name,legend,color in [('heat','Teljes fűtési hő','#CB5555'),('cool','Hűtési hő','#3182BD'),
                               ('aux_heat','Egyéb fűtési hőforrás','#64748B')]:
        if name in frame:axes[2].step(x,col(frame,name),where='post',label=legend,color=color,alpha=.8)
    axes[2].set_ylabel('Teljesítmény [kW]');axes[2].legend(ncol=4,fontsize=8)
    if 'hp_participation_factor' in frame:
        pf=axes[2].twinx();pf.plot(x,col(frame,'hp_participation_factor'),color='#8C6DB0',ls=':',lw=1)
        pf.set(ylabel='Participation factor',ylim=(-.05,1.05))
    energy=col(frame,'bess_energy_kwh',True)
    if np.max(energy)>EPS:
        axes[3].plot(x,energy,label='BESS-energia',color='#8C6DB0')
        power=axes[3].twinx()
        power.step(x,charge,where='post',label='Töltés',color='#E08145')
        power.step(x,col(frame,'p_bess_discharge_kw',True),where='post',label='Kisütés',color='#369277')
        power.set_ylabel('Teljesítmény [kW]');power.legend(loc='upper right',fontsize=8)
        axes[3].legend(loc='upper left',fontsize=8)
    else:axes[3].text(.5,.5,'Nincs aktív BESS ebben az esetben',ha='center',transform=axes[3].transAxes)
    axes[3].set(ylabel='BESS-energia [kWh]',xlabel='Idő')
    for ax in axes:style(ax);blocked_shading(ax,x,settings)
    fig.suptitle('Pozitív: források; negatív: fogyasztás/eladás. Szürke sáv: HP-korlátozási időszak.',fontsize=10)
    dates(axes[-1]);save(fig,out,f'reszletes_{safe(label)}_{safe(key)}_{safe(season)}',dpi)
    return max_error


def battery_day(series,key,out,dpi):
    ordered=[BATTERY_CASE]+[label for label in series if label!=BATTERY_CASE]
    for label in ordered:
        if label not in series:continue
        frame=series[label]
        if 'bess_energy_kwh' not in frame:continue
        energy=frame['bess_energy_kwh']
        swing=energy.resample('D').max()-energy.resample('D').min()
        active=swing[swing>EPS]
        if active.empty:continue
        if BATTERY_DAY is None:day=(active-active.median()).abs().idxmin()
        else:day=pd.Timestamp(BATTERY_DAY)
        chunk=window(frame,day,1)
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        x=chunk.index
        axes[0].plot(x,col(chunk,'p_pv_kw'),label='PV',color='#D6A539')
        axes[0].plot(x,col(chunk,'p_base_kw')+col(chunk,'p_hp_el_kw')-col(chunk,'p_hp_geo_kw'),
                     label='A-mérős alap + HP',color='#64748B')
        axes[0].set(ylabel='Teljesítmény [kW]',title=f'{label} | {key} | BESS-nap: {day:%Y-%m-%d}')
        axes[1].plot(x,col(chunk,'bess_soc'),color='#8C6DB0',label='SOC')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylabel('SOC')
        axes[2].step(x,col(chunk,'p_bess_charge_pv_kw',True),where='post',label='PV-töltés',color='#E08145')
        axes[2].step(x,col(chunk,'p_bess_discharge_kw',True),where='post',label='Kisütés',color='#369277')
        axes[2].step(x,col(chunk,'p_bess_maintenance_charge_kw',True),where='post',label='Veszteségpótló külső töltés',color='#3182BD')
        axes[2].set(ylabel='Teljesítmény [kW]',xlabel='Idő')
        for ax in axes:style(ax);ax.legend(fontsize=8,ncol=3)
        dates(axes[-1]);save(fig,out,f'tipikus_bess_nap_{safe(key)}',dpi)
        chunk.to_csv(out/f'tipikus_bess_nap_{safe(key)}.csv')
        return
    print('[INFO] A kiválasztott háztartásnak nincs aktív BESS-napja az elkészült esetekben')


def solver_report(root,out,dpi):
    records=[]
    for label in CASES:
        for path in sorted((root/label).glob('????-??/solver.json')):
            metadata=json.loads(path.read_text(encoding='utf-8-sig'))
            entries=metadata.get('households',[metadata])
            for entry in entries:
                if 'runtime_s' in entry:
                    records.append(dict(case=label,month=path.parent.name,
                                        household=entry.get('household','közösség'),
                                        status=entry.get('status'),runtime_s=entry['runtime_s'],
                                        mip_gap=entry.get('mip_gap'),bill_ft=entry.get('bill_ft')))
    if not records:return
    frame=pd.DataFrame(records);frame.to_csv(out/'solver_diagnostics.csv',index=False)
    grouped=frame.groupby(['month','case'])['runtime_s'].sum().unstack('case')
    ax=grouped.plot.bar(figsize=(12,5),color=[COLORS.get(c,'#64748B') for c in grouped.columns])
    ax.set(ylabel='Összes solveridő [s]',xlabel='Hónap',title='Solveridő: egyéni modelleknél a háztartások összege')
    style(ax);save(ax.figure,out,'solver_futasidok',dpi)


def community_seasons(cases,out,days,dpi):
    selected=[label for label in cases if '-K' in label]
    if not selected:return
    year=next(iter(cases.values()))['series'].index[0].year
    for season,date in SEASONS:
        fig,axes=plt.subplots(len(selected),1,figsize=(15,3.2*len(selected)),sharex=True,
                              squeeze=False,layout='constrained')
        for ax,label in zip(axes[:,0],selected):
            frame=window(cases[label]['series'],f'{year}-{date}',days)
            shared=col(frame,'p_shared_kw');grid=col(frame,'p_grid_import_total_kw')
            export=col(frame,'p_pv_export_kw')
            ax.stackplot(frame.index,shared,grid,step='post',colors=['#369277','#3182BD'],
                         labels=['REC-vételezés','Hálózati import, GEO-val'],alpha=.8)
            if np.max(export)>EPS:
                ax.fill_between(frame.index,0,-export,where=export>EPS,step='post',color='#CB5555',
                                alpha=.5,label='Hálózati export')
                ax.step(frame.index,np.where(export>EPS,-export,np.nan),where='post',color='#CB5555',lw=1.8)
            ax.axhline(0,color='#183A50',lw=.7)
            ax.set(ylabel='Teljesítmény [kW]',title=SHORT.get(label,label))
            style(ax);ax.legend(ncol=3,fontsize=8)
        axes[-1,0].set_xlabel('Idő');dates(axes[-1,0])
        fig.suptitle(f'{season}: közösségi vételezés (+) és hálózati export (-)',fontsize=12)
        save(fig,out,f'kozossegi_forgalom_{safe(season)}',dpi)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=RESULTS_ROOT)
    parser.add_argument('--out',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--household',default=HOUSEHOLD)
    parser.add_argument('--window-days',type=float,default=WINDOW_DAYS)
    parser.add_argument('--dpi',type=int,default=DPI)
    parser.add_argument('--no-details',action='store_true')
    args=parser.parse_args(argv)
    if not 0<args.window_days<=31 or args.dpi<=0:raise ValueError('Érvényes ablak (0-31 nap) és pozitív DPI szükséges')
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10})
    cases=load_cases(args.results_root)
    annual_plots(cases,args.out,args.dpi)
    community_seasons(cases,args.out,args.window_days,args.dpi)
    key=household_key(cases,args.household)
    series=household_series(cases,key)
    seasonal_comparison(series,key,args.out,args.window_days,args.dpi)
    if MAKE_DETAILS and not args.no_details:
        settings_path=args.results_root/'settings.json'
        settings=json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.is_file() else {}
        checks=[]
        for label in DETAILED_CASES:
            if label not in series:continue
            year=series[label].index[0].year
            for season,date in SEASONS:
                chunk=window(series[label],f'{year}-{date}',args.window_days)
                error=detailed_plot(chunk,label,key,season,args.out,args.dpi,settings)
                checks.append(dict(case=label,household=key,season=season,max_balance_error_kw=error))
        pd.DataFrame(checks).to_csv(args.out/'balance_checks.csv',index=False)
    battery_day(series,key,args.out,args.dpi)
    solver_report(args.results_root,args.out,args.dpi)
    print(f'Kész: {args.out.resolve()} | háztartás: {key} | éves esetek: {list(cases)}')


if __name__=='__main__':
    main()
