cd task

SUFFIX=0518

echo "SUFFIX=$SUFFIX"

# delta = 0

python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling02

python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling02

python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=0 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02

# CFER change delta = 0, 2, 4
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02

python patrolling_task.py --T=10 --tau=15 --delta=2 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=4 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling02

python patrolling_task.py --T=10 --tau=15 --delta=2 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=15 --delta=4 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02

python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling02


