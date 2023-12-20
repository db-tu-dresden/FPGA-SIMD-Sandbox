
print_usage(){
    printf "Hashing project"
    printf "\n\t-v enables AVX512"
    printf "\n\t-c deletes the build folder"
    printf "\n\t-h shows this help\n"
}

clean(){
    printf "Deleting all project files\n"
    rm -r build/
    printf "Deletion done\n"
}

# for now we pull the v_flag out to make sure we compile the project correctly
v_flag="USE_AVX512=ON"
while getopts 'vch' flag; do
    case "${flag}" in
    v)  
        echo "${v_flag}";;
    c) clean
        exit 1 ;;
    *) print_usage
        exit 1 ;;
    esac
done


mkdir -p build
cd build

if test -z "$v_flag"
then
    cmake ..
else
    cmake -D "$v_flag" .. 
fi
cmake --build .