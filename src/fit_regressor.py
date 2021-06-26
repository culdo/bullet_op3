from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from joblib import dump, load

goal_range_bound = np.array([0.11, 0.16, 0.09])
# dynamic_bias = np.array([0.0, 0.02, -0.17])
origin_bias = np.array([0.06, 0.02, -0.17])
sho_origin = (0.0008987878494860629, 0.0695068646038728, 0.38605321951960037)
svr = MultiOutputRegressor(svm.SVR())
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
pybullet_offset = [0, 0, -0.06625]

def gen_train():
    answers = []
    np.random.seed(1)
    for i in range(20):
        rand = np.random.rand(3) * 6.28 - 3.14
        if i in [1, 3, 10]:
            point_info = [(rand * goal_range_bound) + sho_origin + origin_bias]

    np.save("data/train_data.npy", answers)


def gen_predict():
    # train
    data = np.load("data/arm_data.npy")
    y = data[:20, 0]
    X = data[:20, 1]
    svr.fit(X, y)
    dtr.fit(X, y)
    rfr.fit(X, y)
    dump([svr, dtr, rfr], "reg_models.joblib")

    answers = []
    np.random.seed(2)
    for i in range(20):
        rand = np.random.rand(3)
        if i in [1, 3, 10]:
            point_info = (rand * goal_range_bound) + sho_origin + origin_bias + pybullet_offset
            x = [point_info]
            answer = [point_info,
                      svr.predict(x).flatten(),
                      dtr.predict(x).flatten(),
                      rfr.predict(x).flatten()]
            answers.append(answer)
    np.save("data/answers.npy", answers)


gen_predict()
