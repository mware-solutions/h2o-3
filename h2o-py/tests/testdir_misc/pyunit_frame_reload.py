import h2o
from h2o.exceptions import H2OResponseError
from tests import pyunit_utils
import tempfile


def test_frame_reload():
    work_dir = tempfile.mkdtemp()
    iris = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))
    df_key = iris.key
    df_pd_orig = iris.as_data_frame()
    iris.save(work_dir)
    try:
        iris.save(work_dir, force=False)  # fails because file exists
    except H2OResponseError as e:
        assert e.args[0].exception_msg.startswith("File already exists")
    try:
        h2o.load_frame(df_key, work_dir, force=False)  # fails because frame exists
    except H2OResponseError as e:
        assert e.args[0].exception_msg == "Frame Key<Frame> iris_wheader.hex already exists."
    df_loaded_force = h2o.load_frame(df_key, work_dir) 
    h2o.remove(iris)
    df_loaded = h2o.load_frame(df_key, work_dir, force=False)
    df_pd_loaded_force = df_loaded_force.as_data_frame()
    df_pd_loaded = df_loaded.as_data_frame()
    assert df_pd_orig.equals(df_pd_loaded_force)
    assert df_pd_orig.equals(df_pd_loaded)


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_frame_reload)
else:
    test_frame_reload()
