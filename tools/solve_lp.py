from ortools.init.python import init
from ortools.linear_solver import pywraplp
import pandas as pd
from core.common import wdisplay,case_when
import re
import janitor
import typer

app = typer.Typer()

@app.command()
def run(
    inflow_estimate: str = typer.Option("min", help="Specify which inflow estimate to use: 'min' or 'max'"),
    revenue_estimate: str = typer.Option("low", help="Specify which revenue estimate to use: 'low' or 'high'")
    ):
    print("Google OR-Tools version:", init.OrToolsVersion.version_string())
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("NO SOLVER!")
        return False

    revkey=f'net_revenue_ton_{revenue_estimate}'

    # Define the variables, constraints, and objective function here
    df=pd.read_csv('rdrs_geocodes_crindt - SAMPLE DATA LONG.csv')
    df_cap=(
        pd.read_csv('rdrs_geocodes_crindt - rdrs_capacity.csv')
        .clean_names()
        .assign(
            capacity_model=lambda x: case_when(
                x.capacity_amt == 'tons', 1,
                x.capacity_amt == 'cubic yards', 0.4,
                x.capacity_amt == 'gallons', 0.0042,  # Assuming 1 gallon = 0.0042 tons
                True, 1  # Default to 1 ton for any other case
            ) * case_when(
                x.capacity_per == 'day', 365,
                x.capacity_per == 'week', 52,
                x.capacity_per == 'month', 12,
                x.capacity_per == 'year', 1
            ) * x.capacity_model_raw
        )
    )

    df_use = (
        df
        .join(df_cap.set_index('rdrs_id'), on='o_rdrsid')
        .filt(lambda x: ~(x.geometry_merged_orig.isna() | x.geometry_merged_orig.isna() | x.capacity_model.isna()))
        # .filt(lambda x: x.o_rdrsid=='RD10007')
    )

    df_pedro=pd.read_csv('rdrs_geocodes_crindt - MRF data (pedro).csv',skiprows=1,skip_blank_lines=True).clean_names()
    mats=['msw','paper','cardboard','glass','pet','ferrous','mixed_metal','hdpe','mixed_plastic','non_ferrous','other']
    df_pedro.columns=[z+'_out' if z in mats else z for z in [re.sub(r'_2','_in_max',y) for y in [re.sub(r'_1','_in_min',x) for x in df_pedro.columns]]]
    long_cols=[x+'_out' for x in mats] + list(filter(re.compile('(.*_in|out_)').match, df_pedro.columns))
    df_pedro=df_pedro.melt(id_vars=[col for col in df_pedro.columns if col not in long_cols],
        value_vars=long_cols,
        var_name='material_dir',
        value_name='flow'
    )
    df_pedro[['material', 'dir']] = df_pedro['material_dir'].str.split('_', n=1, expand=True)

    df_use=(
        df_use.join(df_pedro.filt(lambda x: x.dir==f'in_{inflow_estimate}').set_index(['rdrs','material'])[['flow']],on=['o_rdrsid','material'])
        .rename(columns={'flow': f'flow_in'})

        # FIXME: hack for first deploy, needs to be resolved
        .filt(lambda x: ~x.flow_in.isna())

        # FIXME: CONVERT TO FLOAT
        .assign(flow_in=lambda x: x.flow_in.astype(float))
    )
    wdisplay(df_use)

    infinity = solver.infinity()
    def create_flow_var(row):
        v = solver.NumVar(0, row.flow_in, f"flow_{row.o_rdrsid}_{row.d_rdrsid}_{row.material}")
        print(v)
        return v
    
    dfx=(
        # flow vars
        df_use.assign(
            f_srk=lambda x: x.apply(create_flow_var, axis=1)
        )
    )
    print("Number of variables =", solver.NumVariables())

    # conservation constraints
    cons_constr=[]
    for [mrf,mat],mrf_mat in dfx.groupby(['o_rdrsid','material']):
        # Create a linear constraint, In_sk = sum_r: f_srk
        #                           : In_sk - sum_r: f_srk = 0
        inflow=mrf_mat.flow_in.max()
        constraint = solver.Constraint(inflow,inflow, f"cons_{mrf}_{mat}")
        for row in mrf_mat.itertuples():
            # Add the variable to the constraint
            constraint.SetCoefficient(row.f_srk, 1)
        cons_constr.append(
            [mrf,mat,constraint]
        )
    cons_constr=pd.DataFrame(cons_constr, columns=['mrf_rdrsid','material','constraint'])

    # capacity constraints
    cap_constr=[]
    for [dest,mat],dest_mat in dfx.groupby(['d_rdrsid','material']):
        constraint = solver.Constraint(0, dest_mat.capacity_model.max(), f"cap_{dest}_{mat}")
        vars = []
        for row in dest_mat.itertuples():
            constraint.SetCoefficient(row.f_srk, 1)
            vars.append(row.f_srk)
        print(vars)
        print(f"SETTING {'+ '.join([str(v) for v in vars])} <= {dest_mat.capacity_model.max()}")
        cap_constr.append(
            [dest,mat,constraint]
        )
    cap_constr=pd.DataFrame(cap_constr, columns=['d_rdrsid','material','constraint'])

    # max_f_srk: profit@s = sum_srk [R_srk - TC_srk]*f_srk
    # here, R_srk would be zero or negative for landfills.
    #
    # s.t.
    # In_sk = sum_r: f_srk
    # diversion rate constraints are simply
    # sum_r f_srk / In_sk <=> diversion rate bounds
    # capital costs could be incorporated as a function of sum_r (f_srk) for each sk using "chunky" constraints for the count of equipment needed to process at least X amount of k at s. This would require a MILP, but you could say things like:
    # ec_ik E_iks >= sum_r(f_srk) for each s
    # where ec_ik is the processing capacity of equipment type i for material k and E_ks is an integer variable indicating the number of ik-type machines installed at s.    


    # Create the objective function, sum_srk [R_srk - TC_srk]*f_srk
    objective = solver.Objective()
    vars = []
    for row in dfx.itertuples():
        print('kk',row)
        objective.SetCoefficient(row.f_srk, getattr(row, revkey))
        vars.append(f'({getattr(row, revkey)}) * {str(row.f_srk)}')
    print(f'Objective function: {" + ".join(vars)}')
    objective.SetMaximization()


    print(f"Solving with {solver.SolverVersion()}")
    result_status = solver.Solve()

    print(f"Status: {result_status}")
    if result_status != pywraplp.Solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if result_status == pywraplp.Solver.FEASIBLE:
            print("A potentially suboptimal solution was found")
        else:
            print("The solver could not solve the problem.")
            return

    print("Solution:")
    for row in dfx.itertuples():
        print(f"Flow variable {row.f_srk} = {row.f_srk.solution_value()}")

    print("Revenue by facility:")
    dfx = dfx.assign(
        solflow=lambda x: x.apply(lambda row: row.f_srk.solution_value(), axis=1),
        revenue=lambda x: x.apply(lambda row: row.f_srk.solution_value() * row[revkey], axis=1)
    )
    for mrf,df in dfx.groupby('o_rdrsid'):
        print(f"MRF {mrf} ({df.flow_in.sum()})....")
        for mat,dfmat in df.filt(lambda x: x.solflow>0).groupby('material'):
            print(f"   Material {mat} ({dfmat.flow_in.sum()})....")
            for row in dfmat.itertuples():
                print(f"      Flow/Rev to {row.d_rdrsid} = {row.solflow} => ${row.revenue}")
            print(f"                MAT TOTAL = {dfmat.solflow.sum()} => ${dfmat.revenue.sum():.2f} (total revenue)")

        print(f"   MRF TOTAL = {df.solflow.sum()} => ${df.revenue.sum():.2f} (total revenue)")

    print("Objective value =", objective.Value())

    print("Advanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")

    dfx.to_csv(f'solution_output_{inflow_estimate}_{revenue_estimate}.csv',index=False)


if __name__ == "__main__":
    init.CppBridge.init_logging("solve_lp.py")
    cpp_flags = init.CppFlags()
    cpp_flags.stderrthreshold = True
    cpp_flags.log_prefix = False
    init.CppBridge.set_flags(cpp_flags)
    app()
