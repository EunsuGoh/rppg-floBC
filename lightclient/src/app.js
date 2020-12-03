import * as exonum from 'exonum-client'
import * as proto from './proto'
import fetchPythonWeights from './utils/fetchPythonWeights';
import fetchDatasetDirectory from './utils/fetchDatasetDirectory';
import fetchClientKeys from './utils/fetchClientKeys';
import { fetchLatestModelTrainer } from './utils/fetchLatestModel';
import store_encoded_vector, { clear_encoded_vector } from './utils/store_encoded_vector'

const INTERVAL_DURATION = 5000

let TRAINER_KEY

fetchClientKeys()
.then((client_keys) => {
  TRAINER_KEY = client_keys
});

function trainNewModel(modelWeights){
    const explorerPath = 'http://127.0.0.1:9000/api/explorer/v1/transactions'

    require("regenerator-runtime/runtime");

    // Numeric identifier of the machinelearning service
    const SERVICE_ID = 3

    // Numeric ID of the `TxShareUpdates` transaction within the service
    const SHAREUPDATES_ID = 0

    let dataset_directory = fetchDatasetDirectory();
    
    fetchPythonWeights(dataset_directory, modelWeights, (model_weights) => {
        clear_encoded_vector();

        const ShareUpdates = new exonum.Transaction({
        schema: proto.TxShareUpdates,
        serviceId: SERVICE_ID,
        methodId: SHAREUPDATES_ID,
        })


        const shareUpdatesPayload = {
        gradients: model_weights,
        seed: exonum.randomUint64(),
        }

        const transaction = ShareUpdates.create(shareUpdatesPayload, TRAINER_KEY)
        const serialized = transaction.serialize()
        console.log(serialized)

        exonum.send(explorerPath, serialized, 10, 3000)
        .then((obj) => console.log(obj))
        .catch((obj) => console.log(obj))
    });
}

setInterval(() => {
    fetchLatestModelTrainer()
    .then(newModel => {
        if(newModel !== -1){
            console.log("New model fetched")
            store_encoded_vector(newModel).then((newModel_path) => {
                trainNewModel(newModel_path)
            });
        }
        else console.log("No New model to fetch, will retry in a bit")
    })
}, INTERVAL_DURATION)

