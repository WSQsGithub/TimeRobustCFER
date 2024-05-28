cd task

python plot_curve_0420.py --dir=../data/patrolling/ --set=tau10_delta0_grid5 --suffix=0518
python plot_curve_0420.py --dir=../data/patrolling/ --set=tau15_delta0_grid5 --suffix=0518
python plot_curve_0420.py --dir=../data/patrolling/ --set=tau15_delta0_grid10 --suffix=0518
python plot_curve_0420.py --dir=../data/patrolling/ --set=tau20_delta0_grid10 --suffix=0518

python plot_sat_0420.py --dir=../data/patrolling/ --set=tau10_delta0_grid5 --suffix=0518
python plot_sat_0420.py --dir=../data/patrolling/ --set=tau15_delta0_grid5 --suffix=0518
python plot_sat_0420.py --dir=../data/patrolling/ --set=tau15_delta0_grid10 --suffix=0518
python plot_sat_0420.py --dir=../data/patrolling/ --set=tau20_delta0_grid10 --suffix=0518
