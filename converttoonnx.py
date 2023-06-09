import joblib
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

with open('iris_model.pkl', 'rb') as f:
    model = joblib.load(f)
    
import sklearn
import sklearn.datasets
import onnx as onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
    
print("Model saved")
