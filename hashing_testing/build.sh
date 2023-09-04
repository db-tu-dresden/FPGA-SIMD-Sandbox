
print_usage(){
    printf "Hashing project"
    printf "-v enables AVX512"
    printf "-c deletes the build folder"
    printf "-h shows this help"
}

clean(){
    rm -r build/
}

while getopts 'vhc' flag; do
    case "${flag}" in
    v) v_flag="USE_AVX512=ON" 
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
cmake --build . --verbose