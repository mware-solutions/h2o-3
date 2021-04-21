package hex.tree.xgboost;

import water.H2O;
import water.Iced;
import water.util.IcedHashMapGeneric;
import water.util.TwoDimTable;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Iced Wrapper around Booster parameter map. The main purpose is to avoid mistakes when using the parameter
 * object directly: this class ensures that returned parameters will be localized.
 */
public class BoosterParms extends Iced<BoosterParms> {

  private IcedHashMapGeneric.IcedHashMapStringObject _parms;

  public static BoosterParms fromMap(Map<String, Object> map) {
    BoosterParms bp = new BoosterParms();
    bp._parms = new IcedHashMapGeneric.IcedHashMapStringObject();
    bp._parms.putAll(map);
    return bp;
  }

  /**
   * @return localized Booster parameters
   */
  public Map<String, Object> get() {
    Map<String, Object> params = localizeDecimalParams(_parms);
    addNodeSpecificParams(params);
    return Collections.unmodifiableMap(params);
  }

  private static void addNodeSpecificParams(final Map<String, Object> params) {
    final String sysGpuId = H2O.getSysProperty("xgboost.gpu.id", null);
    if (sysGpuId != null) {
      params.put("gpu_id", sysGpuId);
    }
  }

  /**
   * Iterates over a set of parameters and applies locale-specific formatting
   * to decimal ones (Floats and Doubles).
   *
   * @param params Parameters to localize
   * @return Map with localized parameter values
   */
  private static Map<String, Object> localizeDecimalParams(final Map<String, Object> params) {
    Map<String, Object> localized = new HashMap<>(params.size());
    for (String key : params.keySet()) {
      final Object value = params.get(key);
      final Object newValue;
      if (value instanceof Float || value instanceof Double) {
        newValue = value.toString();
      } else
        newValue = value;
      localized.put(key, newValue);
    }
    return localized;
  }

  public TwoDimTable toTwoDimTable() {
    final int rows = _parms.size();
    TwoDimTable table = new TwoDimTable(
            "Native XGBoost Parameters", null,
            new String[rows],
            new String[]{"Name", "Value"},
            new String[]{"String", "String"},
            new String[]{"%s", "%s"},
            "");
    int row = 0;
    for (Map.Entry<String, Object> p : _parms.entrySet()) {
      table.set(row, 0, p.getKey());
      table.set(row, 1, p.getValue());
      row++;
    }
    assert row == rows;
    return table;
  }

}
