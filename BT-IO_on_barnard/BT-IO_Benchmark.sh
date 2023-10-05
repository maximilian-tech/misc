#!/bin/env

# This script evaluates the IO performance of a file system
# using the NAS NPB BT-IO Benchmark test.

set -e

NPROCS=(16 64 256 784)
NNODES=( 1  4  16  49)

TOOLCHAIN=(          "foss/2022a"                "foss/2022a"                "intel/2022b" )
TOOLCHAIN_MPIFC=(    "mpif90"                    "mpif90"                    "mpiifort"    )
TOOLCHAIN_MPIFFLAGS=("-fallow-argument-mismatch" "-fallow-argument-mismatch" ""            )
TOOLCHAIN_MPICC=(    "mpicc"                     "mpicc"                     "mpiicc"      )
RUN_ENV_VAR=(        ""                          "export OMPI_MCA_io=\^ompio" ""           )


echo "Are you on a scratch file system? This benchmark creates ~ 140 GB data! Press 'ENTER' to continue"
read

BENCHMARK="NPB3.4.2"

if [ ! -f "${BENCHMARK}.tar.gz" ]; then
    echo "Downloading Benchmark..."
    wget --quiet  https://www.nas.nasa.gov/assets/npb/${BENCHMARK}.tar.gz
fi
if [ ! -d "${BENCHMARK}" ]; then
    echo "Extracting Benchmark..."
    tar -xzf ${BENCHMARK}.tar.gz
fi

echo "Benchmak available"

# SBATCH Template definition

cat << EOF > run.sbatch.tmpl
#!/bin/bash

#SBATCH -A zihforschung
#SBATCH -n xCORESx
#SBATCH -N xNODESx
#SBATCH --ntasks-per-node=16
#SBATCH -c 1
##SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
##SBATCH --exclusive
#SBATCH --hint=nomultithread,memory_bound

set -e

ml purge
ml xTOOLCHAINx


sed -e 's|^MPIFC.*|MPIFC = xTOOLCHAIN_MPI_FCx|g' \
    -e 's|^FFLAGS.*|FFLAGS = xTOOLCHAIN_MPI_FFLAGSx|g' \
    -e 's|^MPICC.*|MPICC = xTOOLCHAIN_MPI_CCx|g' \
    ../config/NAS.samples/make.def.gcc_mpich \
    > ../config/make.def

pushd ../ || exit 1

make BT CLASS=D SUBTYPE=full
make BT CLASS=D SUBTYPE=epio
make BT CLASS=C SUBTYPE=simple

popd 

xRUN_ENV_VARx

srun ../bin/bt.D.x.ep_io         | tee log.bt.D.x.ep_io
rm -f btio.*
srun ../bin/bt.D.x.mpi_io_full   | tee log.bt.D.x.mpi_io_full
rm -f btio.*
srun ../bin/bt.C.x.mpi_io_simple | tee log.bt.C.x.mpi_io_simple
rm -f btio.*

EOF

cat << EOF > make_clean.sh
#!/bin/env bash
set -x
rm -rf NPB3.4-MPI_* \
       *.tmpl \
       run_all_benchmarks.sh \
       create_csv.py \
       create_vis.py \
       run_postprocessing.sbatch \
       make_clean.sh
EOF
chmod +x make_clean.sh

echo "#!/bin/env bash" > run_all_benchmarks.sh

# Fill Template

for idx in ${!TOOLCHAIN[@]}; do
    TC=${TOOLCHAIN[$idx]}
    MPIFC=${TOOLCHAIN_MPIFC[$idx]}
    MPIFFLAGS=${TOOLCHAIN_MPIFFLAGS[$idx]}
    MPICC=${TOOLCHAIN_MPICC[$idx]}
    RUN_ENV=${RUN_ENV_VAR[$idx]}

    printf -v id "%02d" $idx    
    runfolder=NPB3.4-MPI_${TC//\//-}_${id}

    cp -r $BENCHMARK/NPB3.4-MPI $runfolder
    
    sed -e 's|xTOOLCHAINx|'${TC}'|g' \
        -e 's|xTOOLCHAIN_MPI_FCx|'${MPIFC}'|g' \
        -e 's|xTOOLCHAIN_MPI_FFLAGSx|'${MPIFFLAGS}'|g' \
        -e 's|xTOOLCHAIN_MPI_CCx|'${MPICC}'|g' \
        -e 's|xRUN_ENV_VARx|'"${RUN_ENV}"'|g' \
    	run.sbatch.tmpl \
	> $runfolder/run.sbatch.tmpl
    	
    for idx2 in ${!NPROCS[@]}; do
	nprocs=${NPROCS[$idx2]}
	nnodes=${NNODES[$idx2]}
	
        printf -v id2 "%05d" $nprocs    
	runfolder2=run_folder_$id2
	mkdir -p ${runfolder}/${runfolder2}

	sed -e 's|xCORESx|'${nprocs}'|g' \
	    -e 's|xNODESx|'${nnodes}'|g' \
	    $runfolder/run.sbatch.tmpl \
	    > $runfolder/$runfolder2/run.sbatch
	echo "id=\$(sbatch --parsable --chdir=$runfolder/$runfolder2 --dependency=afterok:\$id ./$runfolder/$runfolder2/run.sbatch)" \
	    >> run_all_benchmarks.sh	
    done
    rm $runfolder/run.sbatch.tmpl
done
rm run.sbatch.tmpl



cat << EOF > create_csv.py

import csv
import glob
import dataclasses
from dataclasses import dataclass, asdict
import re


@dataclass
class BenchmarkData:
    toolchain: str
    problem_size: str
    io_type: str
    nprocs: int
    throughput: float# MB/s
    total_written: float # MB
    
@dataclass
class Benchmarks:
    benchmarks: list[BenchmarkData]

def grep_from_file(pattern: str, file: str, delim: str, idx: int) -> str:
    with open(file) as f:
        for line in f:
            if re.search(pattern,line):
                return line.split(delim)[idx].strip()
    return ""
    
files = glob.glob("./*/*/log.*")
elem = []

for file in files:
    elem.append(
        BenchmarkData(
            toolchain = file.split("/")[1].split("_",1)[1],    
            problem_size = grep_from_file("Class",file, "=", 1),
            io_type = file.split(".")[-1].strip(),
            nprocs = int(grep_from_file("Active processes",file, "=", 1)),
            throughput = grep_from_file("I/O data rate  ",file, ":", 1),
            total_written = grep_from_file("Total data written ",file, ":", 1),
        )
    )


Benchmark = Benchmarks(elem)
bench_dict = dataclasses.asdict(Benchmark)

with open('data.csv', 'w', newline='') as csvfile:
    fieldnames = ['toolchain', 'problem_size','io_type','nprocs','throughput','total_written']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for elem in bench_dict['benchmarks']:
         writer.writerow(elem)


EOF

cat << EOF > create_vis.py

## Visualise Data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_updated = pd.read_csv('./data.csv')

print(df_updated.head())

# Renaming toolchain values as per the user's request
df_updated['toolchain'] = df_updated['toolchain'].replace({
    'foss-2022a_00': 'foss-2022a_ompio',
    'foss-2022a_01': 'foss-2022a_romio'
})

# Setting up the figure and main axis again
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting the main data on ax1
sns.lineplot(data=df_updated, x="nprocs", y="throughput", hue="toolchain", style="io_type", markers=True, dashes=False, palette="tab10", markersize=10, ax=ax1)

# Setting the y-axis label and main x-axis label
ax1.set_ylabel("Throughput (MB/s)", fontsize=14)
ax1.set_xlabel("Number of Processes (nprocs)", fontsize=14)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid(True, which="both", ls="--", linewidth=0.5)
ax1.legend(title="Toolchain | IO Type", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Adjusting the second x-axis for nodes with the correct order
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xscale("log")
ax2.set_xticks(sorted(df_updated["nprocs"].unique()))
ax2.set_xticklabels(["1", "4", "16", "49"])
ax2.set_xlabel("Number of Nodes", fontsize=14)

# Setting the title for the plot
plt.title("Comparison of BT-IO Throughput: Horse file system", fontsize=16, y=1.15)
plt.tight_layout()
plt.savefig('througput_over_nprocs.pdf')
plt.savefig('througput_over_nprocs.png')


EOF


cat << EOF > run_postprocessing.sbatch
#!/bin/env bash

#SBATCH -A zihforschung
#SBATCH -c 1 
#SBATCH -N 1  
#SBATCH -n 1 
#SBATCH --mem-per-cpu=2G 

set -e

ml purge
ml foss/2022a
ml Python/3.10.4-bare


if [ ! -f ./.venv/bin/activate ] ; then
    python -m venv .venv
    source ./.venv/bin/activate
    pip install --upgrade pip
    pip install matplotlib seaborn pandas
else
    source ./.venv/bin/activate
fi


srun python create_csv.py
srun python create_vis.py

EOF


echo "id=\$(sbatch --parsable  --dependency=afterok:\$id ./run_postprocessing.sbatch)" \
            >> run_all_benchmarks.sh

sed -i '0,/--dependency=afterok:\([^ ]*\)/s///' run_all_benchmarks.sh

echo "Done"
