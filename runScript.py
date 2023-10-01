#!/usr/bin/env python
#Ubend Channel runScript 
#Fall 2023
#Chris Psenica

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
from mpi4py import MPI
from dafoam import PYDAFOAM, optFuncs
from pygeo import *
from pyspline import *
from idwarp import *
from pyoptsparse import Optimization, OPT
import numpy as np
import json
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from pygeo import geo_utils
from funtofem.mphys import MeldThermalBuilder
from mphys.scenario_aerothermal import ScenarioAeroThermal
from dafoam import PYDAFOAM, optFuncs
from pyspline import *

# =============================================================================
# Input Parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="SNOPT")   # which optimizer to use. Options are: IPOPT (default), SLSQP, and SNOPT
parser.add_argument("-task", help="type of run to do", type=str, default="opt") # which task to run. Options are: opt (default), runPrimal, runAdjoint, checkTotals
args = parser.parse_args()
gcomm = MPI.COMM_WORLD

HFL0 = 241.3
HFL_weight = -0.5
CPL0 = 40.26
CPL_weight = 0.5
U = 8.4

# Set the parameters for optimization
daOptionsAero = {
    "solverName": "DASimpleTFoam",
    "useAD": {"mode": "reverse"},
    "designSurfaces": ["ubend", "ubendup"],
    "primalMinResTol": 1e-8,
    "primalMinResTolDiff": 1e8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U, 0.0, 0.0]},
        "useWallFunction": False,
    },
    "objFunc": {

   "PL": {
            "part1": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["inlet"],
                "scale": 1.0,
                "addToAdjoint": True,
            },
            "part2": {
                "type": "totalPressure",
                "source": "patchToFace",
                "patches": ["outlet"],
                "scale": -1.0,
                "addToAdjoint": True,
            },
        },

        "HFX": {
            "part1": {
                "type": "wallHeatFlux",
                "source": "patchToFace",
                "patches": ["ubend"],
                "scale": -1.0,
                "addToAdjoint": True,
            },
        },

        "TMEAN": {
            "part1": {
                "type": "patchMean",
                "source": "patchToFace",
                "patches": ["outlet"],
                "varName": "T",
                "varType": "scalar",
                "component": 0,
                "scale": -1.0,
                "addToAdjoint": True,
            },
        },

        "skewness": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 20.0,
                "metric": "faceSkewness",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },

        "nonOrtho": {
            "part1": {
                "type": "meshQualityKS",
                "source": "boxToCell",
                "min": [-10.0, -10.0, -10.0],
                "max": [10.0, 10.0, 10.0],
                "coeffKS": 1.0,
                "metric": "nonOrthoAngle",
                "scale": 1.0,
                "addToAdjoint": True,
            },
        },
                                             
    },
    "adjStateOrdering": "cell",
    "normalizeStates": {"U": 10.0, "p": 30.0, "nuTilda": 1e-3, "phi": 1.0, "T": 300.0},
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "natural"},
    # Design variable setup
    "designVar": {
        "shapez": {"designVarType": "FFD"},
        "shapeyouter": {"designVarType": "FFD"},
        "shapeyinner": {"designVarType": "FFD"},
        "shapexinner": {"designVarType": "FFD"},
        "shapexouter1": {"designVarType": "FFD"},
        "shapexouter2": {"designVarType": "FFD"},
    },
}
# mesh warping parameters, users need to manually specify the symmetry plane
meshOptions = {
    "gridFile": os.getcwd(),
    "fileType": "OpenFOAM",
    # point and normal for the symmetry plane
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):
        
        # initialize builders
        dafoam_builder = DAFoamBuilder(daOptionsAero, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)


        # add design variable component and promote to top level
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"]) 

        # add mesh component
        self.add_subsystem("mesh_aero", dafoam_builder.get_mesh_coordinate_subsystem())
        #self.add_subsystem("mesh_thermal", dafoam_builder_thermal.get_mesh_coordinate_subsystem())

        # add geometry component
        self.add_subsystem("geometry_aero", OM_DVGEOCOMP(file="./FFD/UBendDuctFFDSym.xyz", type="ffd"))
        #self.add_subsystem("geometry_thermal", OM_DVGEOCOMP(file="./FFD/UBendDuctFFDSym.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization. For no themal (solid) use ScenarioAerodynamic, for thermal (solid) use ScenarioAerothermal 
        # we pass the builder to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario(
            "scenario",
            ScenarioAerodynamic(
                aero_builder=dafoam_builder,
                #thermal_builder=dafoam_builder_thermal,
                #thermalxfer_builder=thermalxfer_builder,
                
            ),
            #om.NonlinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-8, atol=1e-3),
            #om.LinearBlockGS(maxiter=10, iprint=2, use_aitken=True, rtol=1e-6, atol=1e-1),
        )

        # need to manually connect the x_aero0 between the mesh and geometry components
        self.connect("mesh_aero.x_aero0", "geometry_aero.x_aero_in")
        self.connect("geometry_aero.x_aero0", "scenario.x_aero")

        #self.connect("mesh_thermal.x_thermal0", "geometry_thermal.x_thermal_in")
        #self.connect("geometry_thermal.x_thermal0", "scenario.x_thermal")


    def configure(self):
        # initialize the optimization.
        super().configure()

        # add the objective function to the cruise scenario
        self.scenario.aero_post.mphys_add_funcs()
        #self.scenario.thermal_post.mphys_add_funcs()

        # get surface coordinates from mesh component
        points_aero = self.mesh_aero.mphys_get_surface_mesh()
        #points_thermal = self.mesh_thermal.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry_aero.nom_add_discipline_coords("aero", points_aero)
        #self.geometry_thermal.nom_add_discipline_coords("thermal", points_thermal)
        
        # select FFDs to move
        DVGeo = DVGeometry("./FFD/UBendDuctFFDSym.xyz")
        pts = self.geometry_aero.DVGeo.getLocalIndex(0)

        # shapez
        indexList = []
        indexList.extend(pts[7:16, :, -1].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        shapezVAL = self.geometry_aero.nom_addLocalDV(dvName="shapez" , axis = "z" , pointSelect = PS)
        #shapezVAL = self.geometry_thermal.nom_addLocalDV("shapez" , axis = "z")

        # shapeyouter
        indexList = []
        indexList.extend(pts[7:16, -1, :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        #DVGeo.addLocalDV("shapeyouter", lower=-0.02, upper=0.02, axis="y", scale=1.0, pointSelect=PS, config="configyouter")
        shapeyouterVAL = self.geometry_aero.nom_addLocalDV(dvName="shapeyouter" , axis = "y" , pointSelect = PS)

        # shapeyinner
        indexList = []
        indexList.extend(pts[7:16, 0, :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        #DVGeo.addLocalDV("shapeyinner", lower=-0.04, upper=0.04, axis="y", scale=1.0, pointSelect=PS, config="configyinner")
        shapeyinnerVAL = self.geometry_aero.nom_addLocalDV(dvName="shapeyinner" , axis = "y" , pointSelect = PS)

        # shapexinner
        indexList = []
        indexList.extend(pts[7:16, 0, :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        #DVGeo.addLocalDV("shapexinner", lower=-0.04, upper=0.04, axis="x", scale=1.0, pointSelect=PS, config="configxinner")
        shapexinnerVAL = self.geometry_aero.nom_addLocalDV(dvName="shapexinner" , axis = "x" , pointSelect = PS)


        # shapexouter1
        indexList = []
        indexList.extend(pts[9, -1, :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        #DVGeo.addLocalDV("shapexouter1", lower=-0.05, upper=0.05, axis="x", scale=1.0, pointSelect=PS, config="configxouter1")
        shapexouter1VAL = self.geometry_aero.nom_addLocalDV(dvName="shapexouter1" , axis = "x" , pointSelect = PS)

        # shapexouter2
        indexList = []
        indexList.extend(pts[10, -1, :].flatten())
        PS = geo_utils.PointSelect("list", indexList)
        #DVGeo.addLocalDV("shapexouter2", lower=-0.05, upper=0.0, axis="x", scale=1.0, pointSelect=PS, config="configxouter2")
        shapexouter2VAL = self.geometry_aero.nom_addLocalDV(dvName="shapexouter2" , axis = "x" , pointSelect = PS)

       
        # add outputs for the design variables
        self.dvs.add_output("shapez" , val = np.array([0]*shapezVAL))
        self.dvs.add_output("shapeyouter" , val = np.array([0]*shapeyouterVAL))
        self.dvs.add_output("shapeyinner" , val = np.array([0]*shapeyinnerVAL))
        self.dvs.add_output("shapexinner" , val = np.array([0]*shapexinnerVAL))
        self.dvs.add_output("shapexouter1" , val = np.array([0]*shapexouter1VAL))
        self.dvs.add_output("shapexouter2" , val = np.array([0]*shapexouter2VAL))

        
        # connect the design variables to the geometry
        self.connect("shapez", "geometry_aero.shapez")
        self.connect("shapeyouter", "geometry_aero.shapeyouter")
        self.connect("shapeyinner", "geometry_aero.shapeyinner")
        self.connect("shapexinner", "geometry_aero.shapexinner")
        self.connect("shapexouter1", "geometry_aero.shapexouter1")
        self.connect("shapexouter2", "geometry_aero.shapexouter2")

        
        # define the design variables to the top level
        self.add_design_var("shapez" ,lower=-0.04, upper = 0.04, scaler = 25.0)
        self.add_design_var("shapeyouter" ,lower=-0.02, upper=0.02, scaler = 50.0)
        self.add_design_var("shapeyinner" ,lower=-0.04, upper=0.04, scaler = 25.0)
        self.add_design_var("shapexinner" ,lower=-0.04, upper=0.04, scaler = 25.0)
        self.add_design_var("shapexouter1" ,lower=-0.05, upper=0.05, scaler = 20.0)
        self.add_design_var("shapexouter2" ,lower=-0.05, upper=0.05, scaler = 20.0)

        # add objective and constraints
        self.add_objective("scenario.aero_post.TMEAN", scaler = 1.0)
        self.add_constraint("scenario.aero_post.PL", upper = 55 , scaler=1.0)
        self.add_constraint("scenario.aero_post.skewness", lower=0., upper=6.0, scaler = 1.0)
        self.add_constraint("scenario.aero_post.nonOrtho", lower=0, upper=80.0, scaler = 1.0)


# OpenMDAO setup
prob = om.Problem(reports=None)
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="n2.html")

# initialize the optimization function
optFuncs = OptFuncs([daOptionsAero], prob) 

# use pyoptsparse to setup optimization
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer
# options for optimizers
if args.optimizer == "SNOPT":
    optOptions = {
        "Major feasibility tolerance": 1.0e-7,
        "Major optimality tolerance": 1.0e-7,
        "Minor feasibility tolerance": 1.0e-7,
        "Verify level": -1,
        "Function precision": 1.0e-7,
        "Major iterations limit": 100,
        "Major iterations limit": 100000,
        "Nonderivative linesearch": None,
        "Print file": "opt_SNOPT_print.txt",
        "Summary file": "opt_SNOPT_summary.txt",
    }
elif args.optimizer == "IPOPT":
    optOptions = {
        "tol": 1.0e-7,
        "constr_viol_tol": 1.0e-7,
        "max_iter": 50,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "SLSQP":
    optOptions = {
        "ACC": 1.0e-7,
        "MAXIT": 50,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("opt arg not valid!")
    exit(0)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars", "totals"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "opt":
    # run the optimization
    prob.run_driver()
elif args.task == "runPrimal":
    # just run the primal once
    prob.run_model()
elif args.task == "runAdjoint":
    # just run the primal and adjoint once
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "checkTotals":
    # verify the total derivatives against the finite-difference
    prob.run_model()
    prob.check_totals(
        # of=["scenario.aero_post.CD", "scenario.thermal_post.HF"],
        # wrt=["shape"],
        compact_print=False,
        step=1e-2,
        form="central",
        step_calc="abs",
    )
else:
    print("task arg not found!")
    exit(1)