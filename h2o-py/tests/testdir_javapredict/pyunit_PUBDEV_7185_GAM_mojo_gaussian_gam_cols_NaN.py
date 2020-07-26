import sys, os
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from random import randint
import tempfile

def gam_gaussian_mojo():
    h2o.remove_all()
    params = set_params()   # set deeplearning model parameters    
    train = h2o.import_file("/Users/wendycwong/temp/GamMojoGaussianTrainFail.csv")
    test = h2o.import_file("/Users/wendycwong/temp/GamMojoGaussianTestFail.csv")
    dfnames = train.names
    # add GAM specific parameters
    params["gam_columns"] = []
    params["scale"] = []
    count = 0
    num_gam_cols = 3    # maximum number of gam columns
    excludeList = {"response"}
    for cname in dfnames:
        if not(cname == 'response') and (str(train.type(cname)) == "real"):
            params["gam_columns"].append(cname)
            params["scale"].append(0.001)
            count = count+1
 #           excludeList.add(cname)
            if (count >= num_gam_cols):
                break
    excludeList.add(params['gam_columns'][0])
    x = list(set(train.names) - excludeList)

    TMPDIR = tempfile.mkdtemp()
    gamGaussianModel = pyunit_utils.build_save_model_generic(params, x, train, "response", "gam", TMPDIR) # build and save mojo model
    MOJONAME = pyunit_utils.getMojoName(gamGaussianModel._id)
    h2o.download_csv(test, os.path.join(TMPDIR, 'in.csv'))  # save test file, h2o predict/mojo use same file
    pred_h2o, pred_mojo = pyunit_utils.mojo_predict(gamGaussianModel, TMPDIR, MOJONAME)  # load model and perform predict
    h2o.download_csv(pred_h2o, os.path.join(TMPDIR, "h2oPred.csv"))
    print("Comparing mojo predict and h2o predict...")
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)    # make sure operation sequence is preserved from Tomk        h2o.save_model(glmOrdinalModel, path=TMPDIR, force=True)  # save model for debugging

def set_params():
    missingValues = ['MeanImputation']
    missing_values = missingValues[randint(0, len(missingValues)-1)]

    params = {'missing_values_handling': missing_values, 'family':"gaussian"}
    print(params)
    return params


if __name__ == "__main__":
    pyunit_utils.standalone_test(gam_gaussian_mojo)
else:
    gam_gaussian_mojo()
