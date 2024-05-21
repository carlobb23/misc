from gurobipy import *
import gurobipy as gu
import pandas as pd
import os
import time
import random

# Create Dataframes
I_list, T_list, K_list = [1, 2, 3], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3]
I_list1 = pd.DataFrame(I_list, columns=['I'])
T_list1 = pd.DataFrame(T_list, columns=['T'])
K_list1 = pd.DataFrame(K_list, columns=['K'])
DataDF = pd.concat([I_list1, T_list1, K_list1], axis=1)
Demand_Dict = {(1, 1): 2, (1, 2): 1, (1, 3): 0, (2, 1): 1, (2, 2): 2, (2, 3): 0, (3, 1): 1, (3, 2): 1, (3, 3): 1,
               (4, 1): 1, (4, 2): 2, (4, 3): 0, (5, 1): 2, (5, 2): 0, (5, 3): 1, (6, 1): 1, (6, 2): 1, (6, 3): 1,
               (7, 1): 0, (7, 2): 3, (7, 3): 0}

# Generate Alpha
def gen_alpha(seed):
    random.seed(seed)
    alpha = {(i, t): round(random.random(), 3) for i in I_list for t in T_list}
    return alpha

# General Parameter
max_itr = 10
seed = 123

class MasterProblem:
    def __init__(self, dfData, DemandDF, max_iteration, current_iteration, st, st1):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = dfData['I'].dropna().astype(int).unique().tolist()
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.cons_demand_2 = {}
        self.newvar = {}
        self.cons_lmbda = {}
        self.start = st
        self.start1 = st1

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation_i = self.model.addVars(self.nurses, self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation_i')
        self.x_i = self.model.addVars(self.nurses, self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.BINARY, name='x_i')
        self.lmbda = self.model.addVars(self.nurses, self.roster, vtype=gu.GRB.BINARY, lb=0, name='lmbda')

    def generateConstraints(self):
        for i in self.nurses:
            self.cons_lmbda[i] = self.model.addLConstr(1 == gu.quicksum(self.lmbda[i, r] for r in self.rosterinitial), name = "lmb("+str(i)+")")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.motivation_i[i, t, s, r]*self.lmbda[i, r] for i in self.nurses for r in self.rosterinitial) +
                    self.slack[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand_2[t, s] = self.model.addConstr(
                    gu.quicksum(self.x_i[i, t, s, r]*self.lmbda[i, r] for i in self.nurses for r in self.rosterinitial)
                    >= 0.1*self.demand[t, s], "demand2("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand, self.cons_demand_2

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def solveRelaxModel(self):
        self.model.Params.QCPDual = 1
        self.model.Params.NonConvex = 2
        for v in self.model.getVars():
            v.setAttr('vtype', 'C')
        self.model.optimize()

    def getDuals_i(self):
        Pi_cons_lmbda = self.model.getAttr("Pi", self.cons_lmbda)
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def getDuals_xi(self):
        Pi_cons_demand2 = self.model.getAttr("QCPi", self.cons_demand_2)
        return Pi_cons_demand2


    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        for i in self.nurses:
            for t in self.days:
                for s in self.shifts:
                    if (i, t, s) in self.start:
                        self.model.addLConstr(self.motivation_i[i, t, s, 1] == self.start[i, t, s])
        self.model.update()

    def start2(self):
        for i in self.nurses:
            for t in self.days:
                for s in self.shifts:
                    if self.start[i, t, s] > 0:
                        self.x_i[i, t, s, 1].Start = 1
                    else:
                        self.x_i[i, t, s, 1].Start = 0


    def solveModel(self):
        self.model.Params.QCPDual = 1
        self.model.Params.OutputFlag = 0
        self.model.optimize()

    def addColumn(self, index, itr, schedule):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                qexpr.add(schedule[self.nurseIndex, t, s, self.rosterIndex] * self.lmbda[self.nurseIndex, self.rosterIndex], 1)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon
        self.model.update()

    def addLambda(self, index, itr):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        self.newlmbcoef = 1.0
        current_lmb_cons = self.cons_lmbda[self.nurseIndex]
        expr = self.model.getRow(current_lmb_cons)
        new_lmbcoef = self.newlmbcoef
        expr.add(self.lmbda[self.nurseIndex, self.rosterIndex], new_lmbcoef)
        rhs_lmb = current_lmb_cons.getAttr('RHS')
        sense_lmb = current_lmb_cons.getAttr('Sense')
        name_lmb = current_lmb_cons.getAttr('ConstrName')
        newconlmb = self.model.addLConstr(expr, sense_lmb, rhs_lmb, name_lmb)
        self.model.remove(current_lmb_cons)
        self.cons_lmbda[self.nurseIndex] = newconlmb

    def addColumn2(self, index, itr, schedule2):
        self.nurseIndex = index
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr2 = self.model.getQCRow(self.cons_demand_2[t, s])
                qexpr2.add(schedule2[self.nurseIndex, t, s, self.rosterIndex] * self.lmbda[self.nurseIndex, self.rosterIndex], 1)
                rhs2 = self.cons_demand_2[t, s].getAttr('QCRHS')
                sense2 = self.cons_demand_2[t, s].getAttr('QCSense')
                name2 = self.cons_demand_2[t, s].getAttr('QCName')
                newcon2 = self.model.addQConstr(qexpr2, sense2, rhs2, name2)
                self.model.remove(self.cons_demand_2[t, s])
                self.cons_demand_2[t, s] = newcon2
        self.model.update()

    def checkForQuadraticCons(self):
        self.qconstrs = self.model.getQConstrs()
        print("*{:^88}*".format(f"Check for quadratic constraints {self.qconstrs}"))

    def finalObj(self):
        obj = self.model.objval
        return obj

    def printLambdas(self):
        return self.model.getAttr("X", self.lmbda)

    def finalSolve(self):
        self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.model.update()
        self.model.optimize()

class Subproblem:
    def __init__(self, duals_i, duals_ts, duals_xi, dfData, i, M, iteration, alpha):
        itr = iteration + 1
        self.days = dfData['T'].dropna().astype(int).unique().tolist()
        self.shifts = dfData['K'].dropna().astype(int).unique().tolist()
        self.duals_i = duals_i
        self.duals_ts = duals_ts
        self.duals_xi = duals_xi
        self.Max = 5
        self.Min = 2
        self.M = M
        self.alpha = alpha
        self.model = gu.Model("Subproblem")
        self.index = i
        self.itr = itr

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.x = self.model.addVars([self.index], self.days, self.shifts, [self.itr], vtype=gu.GRB.BINARY, name='x')
        self.y = self.model.addVars([self.index], self.days, vtype=gu.GRB.BINARY, name='y')
        self.mood = self.model.addVars([self.index], self.days, vtype=gu.GRB.CONTINUOUS, lb=0, name='mood')
        self.motivation = self.model.addVars([self.index], self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS, lb=0, name='motivation')

    def generateConstraints(self):
        for i in [self.index]:
            for t in self.days:
                self.model.addLConstr(self.mood[i, t] == 1- self.alpha[i, t])
                self.model.addLConstr(quicksum(self.x[i, t, s, self.itr] for s in self.shifts) == self.y[i, t])
                self.model.addLConstr(gu.quicksum(self.x[i, t, s, self.itr] for s in self.shifts) <= 1)
                for s in self.shifts:
                    self.model.addLConstr(
                        self.motivation[i, t, s, self.itr] >= self.mood[i, t] - self.M * (1 - self.x[i, t, s, self.itr]))
                    self.model.addLConstr(
                        self.motivation[i, t, s, self.itr] <= self.mood[i, t] + self.M * (1 - self.x[i, t, s, self.itr]))
                    self.model.addLConstr(self.motivation[i, t, s, self.itr] <= self.x[i, t, s, self.itr])
            for t in range(1, len(self.days) - self.Max + 1):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max)) <= self.Max)
            self.model.addLConstr(self.Min <= quicksum(self.y[i, t] for t in self.days))


    def generateObjective(self):
        self.model.setObjective(
            0 - gu.quicksum(self.motivation[i, t, s, self.itr] * self.duals_ts[t, s] for i in [self.index] for t in self.days for s in self.shifts) -
            - gu.quicksum(self.x[i, t, s, self.itr] * self.duals_xi[t, s] for i in [self.index] for t in self.days for s in self.shifts)- self.duals_i[self.index], sense=gu.GRB.MINIMIZE)

    def getNewSchedule(self):
        return self.model.getAttr("X", self.motivation)

    def getNewSchedule2(self):
        return self.model.getAttr("X", self.x)

    def getStatus(self):
        return self.model.status

    def solveModel(self):
        self.model.Params.OutputFlag = 0
        self.model.optimize()

#### Normal Solving
class Problem:
    def __init__(self, dfData, DemandDF, alpha):
        self.I = dfData['I'].dropna().astype(int).unique().tolist()
        self.T = dfData['T'].dropna().astype(int).unique().tolist()
        self.K = dfData['K'].dropna().astype(int).unique().tolist()
        self.demand = DemandDF
        self.Max = 5
        self.Min = 2
        self.M = 1e6
        self.alpha = alpha
        self.model = gu.Model("Problems")

    def buildModel(self):
        self.t0 = time.time()
        self.generateVariables()
        self.generateConstraints()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.slack = self.model.addVars(self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, name='slack')
        self.motivation = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='motivation')
        self.x = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name='x')
        self.y = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name='y')
        self.mood = self.model.addVars(self.I, self.T, vtype=gu.GRB.CONTINUOUS, lb=0, name='mood')

    def generateConstraints(self):
        for t in self.T:
            for s in self.K:
                self.model.addConstr(
                    gu.quicksum(self.motivation[i, t, s] for i in self.I) + self.slack[t, s] >= self.demand[t, s])
        for i in self.I:
            for t in self.T:
                self.model.addLConstr(self.mood[i, t] == 1 - self.alpha[i, t])
                self.model.addLConstr(quicksum(self.x[i, t, s] for s in self.K) == self.y[i, t])
                self.model.addLConstr(gu.quicksum(self.x[i, t, s] for s in self.K) <= 1)
                for s in self.K:
                    self.model.addLConstr(self.motivation[i, t, s] >= self.mood[i, t] - self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(self.motivation[i, t, s] <= self.mood[i, t] + self.M * (1 - self.x[i, t, s]))
                    self.model.addLConstr(self.motivation[i, t, s] <= self.x[i, t, s])
            for t in range(1, len(self.T) - self.Max + 1):
                self.model.addLConstr(gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max)) <= self.Max)
            self.model.addLConstr(gu.quicksum(self.y[i, t] for t in self.T) >= self.Min)

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.slack[t, s] for t in self.T for s in self.K), sense=gu.GRB.MINIMIZE)

    def solveModel(self):
        self.model.Params.OutputFlag = 0
        self.model.optimize()
        self.t1 = time.time()

    def getTime(self):
        self.time_total = self.t1 - self.t0
        return self.time_total

    def get_final_values(self):
        return {(i, j, k): round(value, 3) for (i, j, k), value in self.model.getAttr("X", self.motivation).items()}

problem = Problem(DataDF, Demand_Dict, gen_alpha(seed))
problem.buildModel()
problem.solveModel()
obj_val_problem = round(problem.model.objval, 3)
time_problem = round(problem.getTime(), 4)
vals_prob = problem.get_final_values()


# Get Starting Solutions
problem_start = Problem(DataDF, Demand_Dict, gen_alpha(seed))
problem_start.buildModel()
problem_start.model.Params.MIPGap = 0.5
problem_start.model.update()
problem_start.model.optimize()
start_values = {}
for i in I_list:
    for t in T_list:
        for s in K_list:
            start_values[(i, t, s)] = problem_start.motivation[i, t, s].x
start_values2 = {}
for i in I_list:
    for t in T_list:
        for s in K_list:
            start_values[(i, t, s)] = problem_start.x[i, t, s].x


#### Column Generation
modelImprovable = True
t0 = time.time()
itr = 0

# Lists
objValHistSP = []
objValHistRMP = []
avg_rc_hist = []

# Build & Solve MP
master = MasterProblem(DataDF, Demand_Dict, max_itr, itr, start_values, start_values2)
master.buildModel()
master.setStartSolution()
master.start2()
master.updateModel()
master.solveRelaxModel()

# Get Duals from MP
duals_i = master.getDuals_i()
duals_ts = master.getDuals_ts()
duals_xi = master.getDuals_xi()

t0 = time.time()
while (modelImprovable) and itr < max_itr:
    # Start
    itr += 1
    print('* Current CG iteration: ', itr)

    # Solve RMP
    master.current_iteration = itr + 1
    master.solveRelaxModel()
    objValHistRMP.append(master.model.objval)

    # Get Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()
    duals_xi = master.getDuals_xi()

    # Solve SPs
    modelImprovable = False
    for index in I_list:
        subproblem = Subproblem(duals_i, duals_ts, duals_xi, DataDF, index, 1e6, itr, gen_alpha(seed))
        subproblem.buildModel()
        subproblem.solveModel()
        opt_val = subproblem.getNewSchedule()
        opt_val_rounded = {key: round(value, 3) for key, value in opt_val.items()}

        status = subproblem.getStatus()
        if status != 2:
            raise Exception("*{:^88}*".format("Pricing-Problem can not reach optimality!"))

        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)
        print("*{:^88}*".format(f"Reduced cost in Iteration {itr}: {reducedCost}"))
        if reducedCost < -1e-6:
            Schedules = subproblem.getNewSchedule()
            Schedules2 = subproblem.getNewSchedule2()
            master.addColumn(index, itr, Schedules)
            master.addColumn2(index, itr, Schedules2)
            master.addLambda(index, itr)
            master.updateModel()
            modelImprovable = True
            print("*{:^88}*".format(f"Reduced-cost < 0 columns found..."))
    master.updateModel()


    avg_rc = sum(objValHistSP) / len(objValHistSP)
    avg_rc_hist.append(avg_rc)
    objValHistSP.clear()
    print("*{:^88}*".format(f"End CG iteration {itr}"))

    if not modelImprovable:
        print("*{:^88}*".format("No more improvable columns found."))


# Solve MP
master.finalSolve()
