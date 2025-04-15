#!/bin/bash
    export ENCAP_SLURM_INSTANCE=0
    export ENCAP_PROCID=$((0 + $SLURM_PROCID))
    cd scripts/0encap_folder/2D_trotter_cnn_expl_trotter/fix_x_rot
    
    # If $ENCAP_PROCID is 0, then the log file is called log
    if [ "$ENCAP_PROCID" == "0" ]
    then
        log="log"
    else
        log="log_$ENCAP_PROCID"
    fi
    echo $log

    echo "Slurm Job Id: $SLURM_JOB_ID" &> $log
    date &>> $log
    echo "host: $(hostname)" &>> $log
    echo "Slurm Instance: 0" &>> $log

    if [ "1" != "1" ]
    then
        echo "Slurm Proc Id: $SLURM_PROCID" &>> $log
    fi
    echo "Encap Proc Id: $ENCAP_PROCID" &>> $log


    echo "scripts/0encap_folder/2D_trotter_cnn_expl_trotter/fix_x_rot/run.py " &>> $log
    echo "" &>> $log
    #(time python -u run.py ) &>> $log && echo  &>> $log without tee for unbuffered output
    bash -c "time python -u  run.py  2>&1 | tee -a /dev/null" &>> $log
    echo  &>> $log
    
