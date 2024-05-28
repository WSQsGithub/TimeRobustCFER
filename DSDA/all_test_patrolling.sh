cd task

SUFFIX=$1

echo "SUFFIX=$SUFFIX"

# delta = 0
# ---------------------------------------------------------------------------------------------------------------------------


python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01


python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=0 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=0 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=0 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01

# delta = 2
# ---------------------------------------------------------------------------------------------------------------------------

python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01


python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=2 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=2 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=2 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01

# delta = 4
# ---------------------------------------------------------------------------------------------------------------------------
python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=5 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=10 --alg=naive --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=5 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=10 --alg=naiveER --suffix="$SUFFIX" --config=patrolling01


python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=5 --alg=CFER --suffix="$SUFFIX" --config=patrolling01

python patrolling_task.py --T=10 --tau=5 --delta=4 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=10 --delta=4 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01
python patrolling_task.py --T=10 --tau=20 --delta=4 --gridsize=10 --alg=CFER --suffix="$SUFFIX" --config=patrolling01