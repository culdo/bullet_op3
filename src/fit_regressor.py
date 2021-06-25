from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

data = np.load("data/arm_data.npy")
y = data[:20, 0]
X = data[:20, 1]
svr = MultiOutputRegressor(svm.SVR())
svr.fit(X, y)
dtr = DecisionTreeRegressor()
dtr.fit(X, y)
rfr = RandomForestRegressor()
rfr.fit(X, y)

goal_range_bound = np.array([0.11, 0.16, 0.09])
# dynamic_bias = np.array([0.0, 0.02, -0.17])
origin_bias = np.array([0.06, 0.02, -0.17])
center_coord = (-0.0009999999999999992, 0.06823, 0.3900981971520425)
answers = []
np.random.seed(2)
for i in range(20):
    rand = np.random.rand(3)
    if i in [1, 3, 10]:
        point_info = [(rand * goal_range_bound) + center_coord + origin_bias]
        answer = [svr.predict(point_info).flatten(),
                  dtr.predict(point_info).flatten(),
                  rfr.predict(point_info).flatten()]
        answers.append([point_info, answer])
np.save("data/answers.npy", answers)
