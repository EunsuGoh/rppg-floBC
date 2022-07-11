import * as exonum from 'exonum-client';
import * as proto from './proto';
import validate_vector from './validate_vector';
import store_encoded_vector, {
  clear_encoded_vector,
} from './store_encoded_vector';
import { fetchLatestModel, fetchMinScore } from './utils';
import { encode } from 'punycode';
const PythonShell = require('python-shell');

// get decoded Tx from python script
async function decodingTx(encoded_model_weight) {
  const options = {
    mode: 'text',
    scriptPath: './tx_validator/',
    args: [encoded_model_weight],
  };
  const result = await new Promise((resolve, reject) => {
    PythonShell.run('decode_tx.py', options, (err, results) => {
      if (err) return reject(err);
      return resolve(results);
    });
  });
  return result;
}

// TODO : 트랜잭션 객체 확인 필요
function validation() {
  let transaction = proto.TxShareUpdates.decode(
    exonum.hexadecimalToUint8Array(process.argv[3])
  );
  console.log('################################################');
  console.log('validation transaction goes here!', transaction);
  console.log('################################################');
  let min_score = transaction.min_score;
  store_encoded_vector(transaction.gradients).then((encoded_vector_path) => {
    validate_vector(encoded_vector_path, process.argv[2], 0, (valid) => {
      clear_encoded_vector();

      if (valid) {
        console.log('VALID');
      } else {
        console.log('INVALID');
      }
    });
  });
}

fetchLatestModel().then((base_model) => {
  fetchMinScore().then((min_score) => {
    var fs = require('fs');
    var text = fs.readFileSync('../../example/' + process.argv[3], {
      encoding: 'utf8',
    });
    text = text.replace('[', '');
    text = text.replace(']', '');
    text = text.split(',');
    text = text.map(Number);
    // delete a file
    fs.unlink('../../example/' + process.argv[3], (err) => {
      if (err) {
        throw err;
      }
    });
    let transaction = proto.TxShareUpdates.decode(text);
    console.log('################################################');
    console.log('Fetch Latest Model transaction goes Here!', transaction);
    console.log('################################################');
    // console.log("First element = ",transaction.gradients[0])
    // console.log("Last element = ",transaction.gradients[transaction.gradients.length-1])
    let val_id = process.argv[4];
    store_encoded_vector(transaction.gradients, 'gradients' + val_id).then(
      (encoded_vector_path) => {
        if (base_model == 0) {
          validate_vector(
            1,
            '',
            encoded_vector_path,
            process.argv[2],
            min_score,
            (results) => {
              clear_encoded_vector('gradients' + val_id);
              clear_encoded_vector('basemodel' + val_id);

              let verdict = results[0];
              let score = results[1];
              console.log(`${verdict.toUpperCase()}:${score}`);
            }
          );
        } else {
          store_encoded_vector(base_model, 'basemodel' + val_id).then(
            (encoded_basemodel_path) => {
              validate_vector(
                0,
                encoded_basemodel_path,
                encoded_vector_path,
                process.argv[2],
                min_score,
                (results) => {
                  clear_encoded_vector('gradients' + val_id);
                  clear_encoded_vector('basemodel' + val_id);

                  let verdict = results[0];
                  let score = results[1];
                  console.log(`${verdict.toUpperCase()}:${score}`);
                }
              );
            }
          );
        }
      }
    );
  });
});
