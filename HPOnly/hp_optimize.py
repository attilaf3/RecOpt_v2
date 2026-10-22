"""Havi HP + BESS MILP egyénileg; MIQCP arányos közösségi megosztással.
Közvetlen Gurobi API; a változó maradékigény szerepel a súlyban.
"""
import numpy as np
from hp_core import thermal_parameters,blocked_at,settle_results,bills,bess_parameters

# Numerikus szabályozási költség, nem energiaszámla és nem tüzelőanyagár.
# A PF=0 esetben is van ára a céltalan hőtermelésnek.
THERMAL_REGULARIZATION_FT_PER_KWH = 0.01


def solve_window(households,start,stop,s,community=False,states=None,used=None,terminal=True):
    import gurobipy as gp
    from gurobipy import GRB
    states={} if states is None else states
    used=np.zeros(len(households)) if used is None else np.asarray(used)
    n=stop-start;k=len(households)
    if n<=0: raise ValueError('Üres optimalizálási ablak')
    if not any(h.hp is not None or (s.include_bess and h.bess is not None) for h in households):
        result=[dict(hp_el=h.fixed_hp[start:stop].copy(),geo=h.fixed_hp[start:stop].copy()) for h in households]
        flows=settle_results(households,result,start,stop,community)
        money,nextused=bills(flows,np.stack([r['geo'] for r in result]),s,used)
        return result,flows,money,states,nextused,dict(status='Direct',bill_ft=float(money.sum()),comfort_penalty=0.)
    with gp.Model('hp_bess') as m:
        m.Params.OutputFlag=int(s.solver_log);m.Params.MIPGap=s.gap
        m.Params.TimeLimit=s.time_limit
        if community: m.Params.NonConvex=2
        d={};u={};hpvars={};bvars={};bounds={};energies={};lows={};cost=gp.LinExpr();comfort=gp.LinExpr()
        thermal_energy=gp.LinExpr()
        for i,h in enumerate(households):
            pv=h.pv[start:stop];base=h.base[start:stop]
            bp=bess_parameters(h,s.dt) if s.include_bess and h.bess is not None else None
            if bp:
                be=m.addVars(n+1,lb=bp['emin'],ub=bp['emax'],name=f'bess_energy_{i}')
                bc=m.addVars(n,ub=bp['power'],name=f'bess_pv_charge_{i}')
                bd=m.addVars(n,ub=bp['power'],name=f'bess_discharge_{i}')
                bg=m.addVars(n,ub=bp['power'],name=f'bess_maintenance_charge_{i}')
                bz=m.addVars(n,vtype=GRB.BINARY,name=f'bess_charge_mode_{i}')
                initial_bess=float(states.get(h.key+'::bess',bp['initial']))
                if not bp['emin']-1e-7<=initial_bess<=bp['emax']+1e-7:
                    raise ValueError(f'{h.key}: hibás átvitt BESS-állapot')
                m.addConstr(be[0]==initial_bess)
                for t in range(n):
                    m.addConstr(bc[t]+bg[t]<=bp['power']*bz[t])
                    m.addConstr(bd[t]<=bp['power']*(1-bz[t]))
                    before=m.addVar(lb=-bp['power']*s.dt/bp['eta_out'],ub=bp['emax'],name=f'bess_pre_{i}_{t}')
                    gap=m.addVar(lb=-bp['emax'],ub=bp['emin']+bp['power']*s.dt/bp['eta_out'])
                    repair=m.addVar(ub=bp['power']*s.dt*bp['eta_in'])
                    m.addConstr(before==bp['retention']*be[t]+s.dt*(bp['eta_in']*bc[t]-bd[t]/bp['eta_out']))
                    m.addConstr(gap==bp['emin']-before)
                    m.addGenConstrMax(repair,[gap],constant=0.)
                    m.addConstr(bg[t]*s.dt*bp['eta_in']==repair)
                    m.addConstr(be[t+1]==before+repair)
                # Csak az utolsó éves ablakban védjük a kezdeti készletet.
                # Hónapváltáskor a tényleges töltöttség kerül továbbadásra.
                if terminal and stop==len(h.pv): m.addConstr(be[n]>=bp['initial'])
                bvars[i]=(be,bc,bd,bg,bp)
            else:
                bc=bd=bg=[0.]*n
            if h.hp is None:
                # GEO profilt külön mérjük, saját PV vagy REC nem látja el.
                electric=[0.]*n
                cost+=float(h.fixed_hp[start:stop].sum()*s.dt*s.geo)
            else:
                a,b,p=thermal_parameters(h,s.dt)
                x=m.addVars(n+1,2,lb=-30,ub=80,name=f'x_{i}')
                initial=states.get(h.key,(p['heat'],p['heat']))
                for j in range(2): m.addConstr(x[0,j]==float(initial[j]))
                qh=m.addVars(n,ub=p['ph'],name=f'heat_{i}')
                qc=m.addVars(n,ub=p['pc'],name=f'cool_{i}')
                electric=m.addVars(n,ub=max(p['ph']/p['cop'],p['pc']/p['eer']),name=f'hpel_{i}')
                local=m.addVars(n,ub=max(p['ph']/p['cop'],p['pc']/p['eer']),name=f'localhp_{i}')
                other=m.addVars(n,ub=max(p['ph']/p['cop'],p['pc']/p['eer']),name=f'otherhp_{i}')
                short=m.addVars(n,ub=110,name=f'short_{i}')
                targets=[]
                for t in range(n):
                    idx=start+t;mode=h.mode[idx];surplus=max(pv[t]-base[t],0)
                    if mode!=1: qh[t].UB=0
                    if mode!=-1: qc[t].UB=0
                    for j in range(2):
                        m.addConstr(x[t+1,j]==gp.quicksum(a[j,l]*x[t,l] for l in range(2))
                            +b[j,0]*float(h.outdoor[idx])+b[j,1]*(float(h.solar[idx])+qh[t]-qc[t]))
                    m.addConstr(electric[t]==float(h.pf[idx])/p['cop']*qh[t]+qc[t]/p['eer'])
                    # Helyi PV és BESS a HP ellátásához. A teljes háztartási
                    # energiaegyenleg és a BESS-korlátok lejjebb érvényesülnek.
                    m.addConstr(local[t]<=(surplus+bd[t] if h.hp_tariff=="A" else 0));m.addConstr(local[t]+other[t]==electric[t])
                    blocked=blocked_at(idx,s) and h.hp_tariff=="A"
                    target=(p['heat']-p['delta']*(surplus>1e-4) if mode==1
                            else p['cool']+p['delta']*(surplus>1e-4) if mode==-1 else p['heat'])
                    targets.append(target)
                    if mode==1:
                        m.addConstr(x[t+1,0]+short[t]>=(p['hemergency'] if blocked else target))
                        # A teljes fűtés (HP + implicit egyéb hőforrás) sem
                        # melegíthet korlátlanul. Passzív túlmelegedés esetén
                        # lazítás marad, hogy a modell megoldható legyen.
                        m.addConstr(x[t+1,0]-short[t]<=p['heat']+p['delta'])
                    elif mode==-1:
                        m.addConstr(x[t+1,0]-short[t]<=(p['cemergency'] if blocked else target))
                        m.addConstr(x[t+1,0]+short[t]>=p['cool']-p['delta'])
                    else: m.addConstr(short[t]==0)
                    if blocked and mode:
                        critical=m.addVar(vtype=GRB.BINARY,name=f'critical_{i}_{t}')
                        emergency=m.addVar(vtype=GRB.BINARY,name=f'emergency_{i}_{t}')
                        if mode==1:
                            m.addGenConstrIndicator(critical,True,x[t,0]<=p['hcritical'])
                            m.addGenConstrIndicator(emergency,True,x[t,0]<=p['hemergency'])
                        else:
                            m.addGenConstrIndicator(critical,True,x[t,0]>=p['ccritical'])
                            m.addGenConstrIndicator(emergency,True,x[t,0]>=p['cemergency'])
                        m.addConstr(local[t]<=max(p['ph']/p['cop'],p['pc']/p['eer'])*critical)
                        # REC is külső forrás a korábbi HP tiltási logikában.
                        m.addConstr(other[t]<=max(p['ph']/p['cop'],p['pc']/p['eer'])*emergency)
                # Hónapvégi levegő- és tömeghőmérséklet cél: ne fogyjon el ingyen a hő.
                if terminal:
                    endmode=h.mode[stop-1]
                    extra=m.addVars(2,ub=110,name=f'endshort_{i}')
                    for j in range(2):
                        if endmode==1: m.addConstr(x[n,j]+extra[j]>=p['heat']-p['delta'])
                        elif endmode==-1: m.addConstr(x[n,j]-extra[j]<=p['cool']+p['delta'])
                        else: m.addConstr(extra[j]==0)
                    comfort+=gp.quicksum(extra[j] for j in range(2))*s.dt
                comfort+=gp.quicksum(short[t] for t in range(n))*s.dt
                thermal_energy+=gp.quicksum(qh[t]+qc[t] for t in range(n))*s.dt
                hpvars[i]=(x,qh,qc,electric,short,np.asarray(targets))
            if h.hp is not None and h.hp_tariff=="GEO":
                cost+=s.geo*s.dt*gp.quicksum(electric[t] for t in range(n))
            maxel=(max(p['ph']/p['cop'],p['pc']/p['eer']) if h.hp is not None else 0)
            for t in range(n):
                net=m.addVar(lb=-float(pv[t]),ub=float(base[t]+maxel),name=f'net_{i}_{t}')
                neg=m.addVar(lb=-float(base[t]+maxel),ub=float(pv[t]),name=f'neg_{i}_{t}')
                predef=m.addVar(ub=float(base[t]+maxel),name=f'predef_{i}_{t}')
                presur=m.addVar(ub=float(pv[t]),name=f'presur_{i}_{t}')
                m.addConstr(net==float(base[t]-pv[t])+(electric[t] if h.hp_tariff=="A" else 0))
                m.addConstr(neg==-net)
                m.addGenConstrMax(predef,[net],constant=0.)
                m.addGenConstrMax(presur,[neg],constant=0.)
                if bp:
                    m.addConstr(bc[t]<=presur)
                    m.addConstr(bd[t]<=predef)
                bounds[i,t]=float(base[t]+maxel)+(bp['power'] if bp else 0.)
                d[i,t]=m.addVar(ub=bounds[i,t],name=f'd_{i}_{t}')
                u[i,t]=m.addVar(ub=float(pv[t]),name=f'u_{i}_{t}')
                m.addConstr(d[i,t]==predef-bd[t]+bg[t])
                m.addConstr(u[i,t]==presur-bc[t])
                if bp:
                    # Veszteségpótló külső töltés mellett nem adunk tovább
                    # saját PV-t: előbb azt kell a tárolóba irányítani.
                    repair_mode=m.addVar(vtype=GRB.BINARY,name=f'repair_mode_{i}_{t}')
                    m.addConstr(bg[t]<=bp['power']*repair_mode)
                    m.addConstr(u[i,t]<=float(pv[t])*(1-repair_mode))
            energy=m.addVar(ub=sum(bounds[i,t] for t in range(n))*s.dt,name=f'energy_{i}')
            low=m.addVar(ub=max(s.annual_cap-used[i],0),name=f'low_{i}')
            m.addConstr(energy==s.dt*gp.quicksum(d[i,t] for t in range(n)))
            m.addGenConstrMin(low,[energy],constant=max(float(s.annual_cap-used[i]),0))
            energies[i]=energy;lows[i]=low
            cost+=s.high*energy+(s.low-s.high)*low
            # Helyi maradék PV-t export vagy REC-eladás veszi át azonos eladói áron.
            cost-=s.export*s.dt*gp.quicksum(u[i,t] for t in range(n))
        if community:
            shares=m.addVars(k,n,ub=max(bounds.values()),name='shared_in')
            for t in range(n):
                td=m.addVar(ub=sum(bounds[i,t] for i in range(k)))
                tu=m.addVar(ub=sum(float(h.pv[start+t]) for h in households))
                total=m.addVar(ub=sum(float(h.pv[start+t]) for h in households))
                m.addConstr(td==gp.quicksum(d[i,t] for i in range(k)))
                m.addConstr(tu==gp.quicksum(u[i,t] for i in range(k)))
                m.addGenConstrMin(total,[td,tu])
                m.addConstr(gp.quicksum(shares[i,t] for i in range(k))==total)
                for i in range(k):
                    m.addConstr(shares[i,t]<=d[i,t])
                    # S_i / S_total = deficit_i / deficit_total. Zéróigénynél
                    # share<=deficit és sum(share)=0 rögzíti a nullát.
                    m.addQConstr(shares[i,t]*td==d[i,t]*total)
            for i in range(k):
                se=m.addVar(ub=sum(bounds[i,t] for t in range(n))*s.dt,name=f'share_energy_{i}')
                sl=m.addVar(ub=max(s.annual_cap-used[i],0),name=f'share_low_{i}')
                m.addConstr(se==s.dt*gp.quicksum(shares[i,t] for t in range(n)))
                m.addConstr(sl<=se);m.addConstr(sl<=lows[i])
                # Az elszámolás havi átlagárát pontosan a solverben is követjük.
                m.addQConstr(sl*energies[i]==se*lows[i])
                cost+=(s.shared_high+s.shared_rhd-s.high)*se
                cost+=((s.shared_low-s.shared_high)-(s.low-s.high))*sl
        m.setObjective(cost+s.penalty*comfort+THERMAL_REGULARIZATION_FT_PER_KWH*thermal_energy,GRB.MINIMIZE)
        m.optimize()
        if m.SolCount==0:
            if m.Status==GRB.INFEASIBLE:
                m.computeIIS();m.write('hp_only_infeasible.ilp')
            raise RuntimeError(f'HP megoldás nincs: Gurobi status={m.Status}')
        if m.ConstrVio>1e-5 or m.BoundVio>1e-5 or m.IntVio>1e-5:
            raise RuntimeError('Gurobi megoldás megengedettségi hibája túl nagy')
        result=[];nextstates={}
        for i,h in enumerate(households):
            if i not in hpvars:
                result.append(dict(hp_el=h.fixed_hp[start:stop].copy(),geo=h.fixed_hp[start:stop].copy()))
                continue
            x,qh,qc,el,short,targets=hpvars[i]
            air=np.array([x[t,0].X for t in range(n+1)])
            mass=np.array([x[t,1].X for t in range(n+1)])
            heat=np.array([qh[t].X for t in range(n)])
            result.append(dict(hp_el=np.array([el[t].X for t in range(n)]),geo=np.zeros(n),air=air,mass=mass,
                heat=heat,cool=np.array([qc[t].X for t in range(n)]),target=targets,
                comfort_shortfall=np.array([short[t].X for t in range(n)]),aux_heat=(1-h.pf[start:stop])*heat))
            if h.hp_tariff=='GEO': result[-1]['geo']=result[-1]['hp_el'].copy()
            nextstates[h.key]=(air[-1],mass[-1])
        for i,h in enumerate(households):
            r=result[i]
            if i in bvars:
                be,bc,bd,bg,bp=bvars[i]
                r.update(bess_energy=np.array([be[t].X for t in range(n+1)]),
                         bess_charge=np.array([bc[t].X for t in range(n)]),
                         bess_discharge=np.array([bd[t].X for t in range(n)]),
                         bess_grid_charge=np.array([bg[t].X for t in range(n)]))
                r['bess_soc']=r['bess_energy']/bp['size']
                r['bess_loss']=(1-bp['retention'])*r['bess_energy'][:-1]
                nextstates[h.key+'::bess']=float(r['bess_energy'][-1])
                if not np.allclose(np.diff(r['bess_energy']),
                    s.dt*(bp['eta_in']*(r['bess_charge']+r['bess_grid_charge'])-r['bess_discharge']/bp['eta_out'])-r['bess_loss'],
                    rtol=1e-5,atol=1e-5): raise RuntimeError(f'{h.key}: hibás BESS-energiaegyenleg')
            else:
                r.update(bess_energy=np.zeros(n+1),bess_soc=np.zeros(n+1),
                         bess_charge=np.zeros(n),bess_discharge=np.zeros(n),
                         bess_grid_charge=np.zeros(n),bess_loss=np.zeros(n))
        flows=settle_results(households,result,start,stop,community)
        if community:
            solver_shares=np.array([[shares[i,t].X for t in range(n)] for i in range(k)])
            if not np.allclose(solver_shares,flows['shared'],rtol=1e-5,atol=1e-5):
                raise RuntimeError('Solver és tényleges arányos megosztás eltér: skálázás/tolerancia ellenőrzése szükséges')
        money,nextused=bills(flows,np.stack([r['geo'] for r in result]),s,used)
        if not np.isclose(money.sum(),cost.getValue(),rtol=1e-6,atol=.1):
            raise RuntimeError('Optimalizáló cél és arányos elszámolás nem egyezik')
        metadata=dict(status=int(m.Status),mip_gap=float(m.MIPGap),runtime_s=float(m.Runtime),
                      objective_with_comfort=float(m.ObjVal),bill_ft=float(money.sum()),
                      comfort_penalty=float(s.penalty*comfort.getValue()),
                      thermal_regularization=float(THERMAL_REGULARIZATION_FT_PER_KWH*thermal_energy.getValue()),
                      thermal_control_version='two_sided_comfort_v1')
    return result,flows,money,nextstates,nextused,metadata
