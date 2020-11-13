//Default weights length
const WEIGHTS_LENGTH = 607394;

export default function parsePythonList(list){
    let in_arr = list.trim().replace(/(\r\n|\n|\r)/gm, "");
    if (in_arr[0] != '[' || in_arr[in_arr.length-1] != ']') {
        console.log("Syntax Error: Python returned a faulty array");
        process.exit();
    }
    let weights = in_arr.slice(1, in_arr.length - 1).split(',').filter((el) => el != "")
    if (weights.length != WEIGHTS_LENGTH){
        console.log("We only support weights of length ", WEIGHTS_LENGTH);
        process.exit();
    }
    for (let i = 0 ; i < WEIGHTS_LENGTH ; i++){
        if (isNaN(weights[i])){
            console.log("Error: ", weights[i], " is not a number");
            process.exit();
        }
        weights[i] = parseFloat(weights[i]);
    }
    return weights;
}
