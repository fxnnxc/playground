#!/bin/bash

getInputFromList(){
        list=$1
        result=0

        while ! [ $result = 1 ]; 
        do
                echo "[INFO] Possible Choices: "$list >&2
                printf "[Input] value : " >&2
                read var
                for item in  $(echo $list | tr " " " ");
                do
                        echo $item
                        if [ $var = $item ]; then
                                printf "$var"
                                result=1
                                return 0

                        fi
                done
        done
}


getArgumentInput(){
        value=$1
        printf "[Input] $value : " >&2
        read -e  var
        printf -- "$var"
}



echo $params
