set -xe

source deactivate
conda activate /nfs_beijing_td/nfsshare/nfshome/ming/tools/miniconda3/envs/ByProt

PROJECT_HOME=$(cd "$(dirname "$0")" && pwd)

exp_path=$1  # ../experiments/protein_mpnn_cmlm
pdb_list=$2  # ../examples/inpaint_test/pdb.list
outfile=$3  # ../examples/inpaint_test/pred.pkl

python ${PROJECT_HOME}/inpaint_protein_mpnn_cmlm.py ${exp_path} ${pdb_list} ${outfile}
