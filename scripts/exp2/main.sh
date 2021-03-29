command_start="sh "
trainers_max=2
validators_max=1
command="./scripts/spawn/spawn.sh "
# while getopts "n:t:cbjl" arg; do
while getopts "n:t:bl" arg; do
    case $arg in
    n) 
        validators_max=$(($OPTARG)) 
        ;;
    # c) 
    #     command+="-c "
    #     ;;
    b) 
        command+="-b "
        ;;
    # j) 
    #     command+="-j "
    #     ;;
    l) 
        command_start="bash "
        ;;
    t) 
        trainers_max="$OPTARG"
        ;;
    esac
done

# start_public_port=1000
# start_peer_port=5000
source ./scripts/utils/newTab.sh
command_string=$command

for ((i=1;i<=trainers_max;i++));
do
    for ((j=1;j<=validators_max;j++));
    do
        echo "Trainers =  $i, Validators = $j"
        command=$command_string
        # openTab $command_start "$command_start $command -n $j -t $i -p ./exp_n${j}_t${i} -w $start_public_port -q $start_peer_port"
        $command_start $command -c -j -n $j -t $i
        sleep 10
        # start_public_port=$((start_public_port+1))
        # start_peer_port=$((start_peer_port+1))
    done
done

