# mpirun -np 8 --allow-run-as-root 
# -x LD_PRELOAD=/opt/rocm/lib/librccl.so
# -x NCCL_DEBUG=WARN 
# -x RCCL_MSCCLPP_THRESHOLD=128 
# -x MSCCL_ALGO_DIR=/mnt/v-feilong/xml_test/ag_allpairs 
# /mnt/v-feilong/rccl-tests/build/all_gather_perf -b 128 -f 2 -g 1 -c 1 -w 20 -G 1 -e 16MB -n 100

# mpirun --allow-run-as-root --tag-output -hostfile /mnt/v-feilong/hostfile -map-by ppr:8:node --bind-to numa 
# -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 
# -x LD_PRELOAD=/opt/rocm/lib/librccl.so:$LD_PRELOAD 
# -x NCCL_DEBUG=WARN 
# -x RCCL_MSCCLPP_THRESHOLD=128 
# -x RCCL_MSCCL_FORCE_ENABLE=1 
# -x RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1 
# -x RCCL_MSCCLPP_ENABLE=0 
# -x MSCCL_ALGO_DIR=/mnt/v-feilong/xml_test/ag_ring 
# /mnt/v-feilong/rccl-tests/build/all_gather_perf -b 128 -f 2 -g 1 -c 1 -w 20 -e 1GB -n 100
 

LD_PRELOAD=/opt/rocm/lib/librccl.so:$LD_PRELOAD
NCCL_DEBUG=WARN
RCCL_MSCCLPP_THRESHOLD=0
RCCL_TEST_OUTPUT_PREFIX="-Z csv --output_file "
RCCL_ADDITIONAL_OPTS=" --timeout 10 "

MPI_COMMON_V="/opt/ompi/bin/mpirun --allow-run-as-root --tag-output" # Verbose caused by "--tag-output"
MPI_COMMON="/opt/ompi/bin/mpirun --allow-run-as-root"
MPI_1_NODE="-np 8"
MPI_2_NODES="-hostfile /mnt/v-feilong/hostfile -map-by ppr:8:node --bind-to numa"

MCA_COMMON="-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0"

PERF_PATH_PRE="/mnt/v-feilong/rccl-tests/build/"

# perf a xml to get cc type
fetch_type() {
    local xml_file=$1
    # <algo name="allgather_allpairs" proto="LL" nchannels="1" nchunksperloop="16" ngpus="16" coll="allgather" inplace="1" outofplace="0" minBytes="0" maxBytes="0">
    # get field in coll:
    local coll_type=$(grep -oP 'coll="\K[^"]+' ${xml_file})

    case ${coll_type} in
        allgather)
            echo "all_gather"
            ;;
        allreduce)
            echo "all_reduce"
            ;;
        reduce)
            echo "reduce"
            ;;
        broadcast)
            echo "broadcast"
            ;;
        reduce_scatter)
            echo "reduce_scatter"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

exec_one_node() {
    local algo_dir=$1
    local algo_type=$2
    local b=$3
    local e=$4

    ${MPI_COMMON} ${MPI_1_NODE} \
    -x LD_PRELOAD=${LD_PRELOAD} \
    -x NCCL_DEBUG=${NCCL_DEBUG} \
    -x RCCL_MSCCLPP_THRESHOLD=${RCCL_MSCCLPP_THRESHOLD} \
    -x MSCCL_ALGO_DIR=${algo_dir} \
    ${PERF_PATH_PRE}${algo_type}_perf \
    -b ${b} -e ${e} \
    -f 2 -g 1 -c 1 \
    -w 20 -n 100 \
    ${RCCL_ADDITIONAL_OPTS}
}

exec_two_nodes() {
    local algo_dir=$1
    local algo_type=$2
    local b=$3
    local e=$4

    ${MPI_COMMON} ${MPI_2_NODES} ${MCA_COMMON} \
    -x LD_PRELOAD=${LD_PRELOAD} \
    -x NCCL_DEBUG=${NCCL_DEBUG} \
    -x RCCL_MSCCLPP_THRESHOLD=${RCCL_MSCCLPP_THRESHOLD} \
    -x RCCL_MSCCL_FORCE_ENABLE=1 \
    -x RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1 \
    -x RCCL_MSCCLPP_ENABLE=0 \
    -x MSCCL_ALGO_DIR=${algo_dir} \
    ${PERF_PATH_PRE}${algo_type}_perf \
    -b ${b} -e ${e} \
    -f 2 -g 1 -c 1 \
    -w 20 -n 100 \
    ${RCCL_ADDITIONAL_OPTS}
}

setup_temp_dir() {
    local xml_path=$1
    local dst_dir_path="/tmp/v-yxli/test/single-ccl/xml/"

    if [ ! -d ${dst_dir_path} ]; then
        mkdir -p ${dst_dir_path}
    fi

    rm ${dst_dir_path}/*.xml
    cp ${xml_path} ${dst_dir_path}/test.xml

    echo ${dst_dir_path}
}

exec_xml() {
    local xml_path=$1
    local lower_bound=$2
    local upper_bound=$3
    
    local rank_num=$(grep -oP 'ngpus="\K[^"]+' ${xml_path})
    local algo_type=$(fetch_type ${xml_path})

    local xml_temp_path=$(setup_temp_dir ${xml_path})

    if [[ ${rank_num} -le 9 ]]; then
        exec_one_node ${xml_temp_path} ${algo_type} ${lower_bound} ${upper_bound}
    else
        exec_two_nodes ${xml_temp_path} ${algo_type} ${lower_bound} ${upper_bound}
    fi
}


main() {
    local set_name=$1
    local input_path="Input/${set_name}"
    local output_path="GroundTruth/${set_name}"
    if [ ! -d ${output_path} ]; then
        mkdir -p ${output_path}
    fi
    local lower=8B
    local upper=4GB

    for xml_file in ${input_path}/*.xml; do
        echo "Executing XML: ${xml_file}"
        local xml_file_name=$(basename ${xml_file})
        local output_file="${output_path}/${xml_file_name%.*}_result.log"
        exec_xml ${xml_file} ${lower} ${upper} > ${output_file}
        echo "Output saved to: ${output_file}"

        awk 'BEGIN {
            OFS=","; 
            header="size,count,type,redop,root,time,algbw,busbw,wrong";
            print header > "'${output_path}/${xml_file_name%.*}'_out_of_place.csv"; 
            print header > "'${output_path}/${xml_file_name%.*}'_in_place.csv"
        }
        /^[[:space:]]*[0-9]/ { 
            # Out-of-place
            print $1,$2,$3,$4,$5,$6,$7,$8,$9 > "'${output_path}/${xml_file_name%.*}'_out_of_place.csv";
            # In-place
            print $1,$2,$3,$4,$5,$10,$11,$12,$13 > "'${output_path}/${xml_file_name%.*}'_in_place.csv" 
        }' ${output_file}

    done
}

main "$1"
