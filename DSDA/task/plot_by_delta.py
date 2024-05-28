# %%
from core.utils import *
import argparse

parser = argparse.ArgumentParser(description='Input tau/gridsize')
parser.add_argument('--dir', type=str, help='directory of the log file, e.g. ../data/patrolling/', default='../data/patrolling/')
parser.add_argument('--alg', type=str, help='algorithm of the experiment, e.g. CFER', default='CFER')
parser.add_argument('--set', type=str, help='setting of the experiment, e.g. tau5_gird5', default='tau5_grid5')
parser.add_argument('--suffix', type=str, help='suffix of the log file', default='0514')


args = parser.parse_args()
# read data

file_dir = args.dir
setting = args.set
suffix = args.suffix

save_curve_dir = '../images_' + suffix +'/reward/'
save_sat_dir = '../images_'+ suffix +'/sat/'

delta_values = ["delta0","delta2", "delta4"]

test_logs = []
train_logs = []

for delta in delta_values:
    # print(prefix)
    filepath = file_dir + args.alg + '_' + setting.split('_')[0] + '_' + delta + '_' + setting.split('_')[1] + '_' + suffix + '.mat'
    train_log, test_log = readLog(filepath)
    # print(test_log)
    test_logs.append(test_log)
    train_logs.append(train_log)
    
colors = ["#C848b9","#F962A7","#FD836D"]

# plot learning curve with plotly

fig, ax = plt.subplots(figsize=(5,5))
window_size = 50

for i in range(3):
    # print(file_paths[i])
    data = getLearningCurveData(train_logs[i])
    # smoothed_arr, lower_bounds, upper_bounds = smoothData(data, 100)
    smoothed_arr, lower_bounds, upper_bounds = curveData(data, window_size)
    # plot smoothed data
    x = np.array(range(len(smoothed_arr)))*window_size
    plt.plot(x, smoothed_arr, c=colors[i],linewidth=2, label=delta_values[i])
    
    plt.fill_between(x, lower_bounds, upper_bounds, facecolor=colors[i], alpha=0.2)

curve_fname = save_curve_dir + file_dir.split('/')[2] +"_"+ setting +"_"+"Delta"+ suffix
plt.legend()
plt.savefig(curve_fname)
print('Figure saved as', curve_fname)
plt.close()

fig, ax = plt.subplots(figsize=(5,5))
window_size = 50

for i in range(3):
    # print(file_paths[i])
    data = getLearningRewardData(train_logs[i])
    # smoothed_arr, lower_bounds, upper_bounds = smoothData(data, 100)
    smoothed_arr, lower_bounds, upper_bounds = curveData(data, window_size)
    # plot smoothed data
    x = np.array(range(len(smoothed_arr)))*window_size
    plt.plot(x, smoothed_arr, c=colors[i],linewidth=2, label=delta_values[i])
    
    plt.fill_between(x, lower_bounds, upper_bounds, facecolor=colors[i], alpha=0.2)

sat_fname = save_sat_dir + file_dir.split('/')[2] + "_" + setting + "_"+ "Delta"+ suffix
plt.legend()
plt.savefig(sat_fname)
print('Figure saved as', sat_fname)
plt.close()

