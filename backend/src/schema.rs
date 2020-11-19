// Copyright 2020 The Exonum Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Cryptocurrency database schema.
use exonum::{
    crypto::{Hash, PublicKey},
    merkledb::{
        access::{Access, FromAccess, RawAccessMut},
        Entry, Group, MapIndex, ProofListIndex, RawProofMapIndex,
    },
    runtime::CallerAddress as Address,
};
use exonum_derive::{FromAccess, RequireArtifact};

// modified
use crate::{model::Model, INIT_WEIGHT, LAMBDA, MODEL_SIZE};
#[path = "model.rs"]

const DEBUG: bool = false;

/// Database schema for the cryptocurrency.
///
/// Note that the schema is crate-private, but it has a public part.
#[derive(Debug, FromAccess)]
pub(crate) struct SchemaImpl<T: Access> {
    /// Public part of the schema.
    #[from_access(flatten)]
    pub public: Schema<T>,
    /// History for specific wallets.
    // modified
    pub model_history: Group<T, u32, ProofListIndex<T::Base, Hash>>,
    /// Trainer scores mapped by their addresses
    pub trainers_scores: MapIndex<T::Base, Address, String>,
}

/// Public part of the cryptocurrency schema.
#[derive(Debug, FromAccess, RequireArtifact)]
#[require_artifact(name = "exonum-ML")]
pub struct Schema<T: Access> {
    /// Map of model keys to information about the corresponding account.
    // modified
    pub models: RawProofMapIndex<T::Base, Address, Model>,
    /// Lastest model Addr
    pub latest_version_addr: Entry<T::Base, Address>,
}

impl<T: Access> SchemaImpl<T> {
    pub fn new(access: T) -> Self {
        Self::from_root(access).unwrap()
    }
}

impl<T> SchemaImpl<T>
where
    T: Access,
    T::Base: RawAccessMut,
{
    // Register a trainer's identity
    pub fn register_trainer(&mut self, trainer_addr: Address) {
        println!("Registering {:?}...", trainer_addr);
        let num_of_trainers = (self.trainers_scores.values().count() + 1) as f64;
        let starter_score: f64 = 1.0 / (LAMBDA * num_of_trainers);
        // Insert new score only if trainer wasn't registered
        if self.trainers_scores.contains(&trainer_addr) == false {
            self.trainers_scores
                .put(&trainer_addr, starter_score.to_string());
        }
        if DEBUG {
            println!("Printing trainer addr / scores:");
            for entry in self.trainers_scores.iter() {
                println!("{:?}", entry);
            }
        }
    }
    // modified
    pub fn update_weights(&mut self, updates: Vec<Vec<f32>>) {
        let mut latest_model: Model;
        let model_values = self.public.models.values();
        if model_values.count() == 0 {
            let version: u32 = 0;
            let version_hash = Address::from_key(SchemaUtils::pubkey_from_version(version));
            latest_model = Model::new(version, MODEL_SIZE, vec![INIT_WEIGHT; MODEL_SIZE as usize]);
            println!("Initial Model: {:?}", latest_model);
            self.public.models.put(&version_hash, latest_model);
            self.public.latest_version_addr.set(version_hash);
        }

        /*
        let model_values2 = self.public.models.values();
        println!("Printing all models:");
        for val in model_values2 {
            println!("{:?}", val);
        }
        */

        let version_hash = self.public.latest_version_addr.get().unwrap();
        latest_model = self.public.models.get(&version_hash).unwrap();
        println!("Latest Model: {:?}", (&latest_model));

        let mut new_model: Model = Model::new(
            (&latest_model).version + 1,
            (&latest_model).size,
            (&latest_model).weights.clone(),
        );
        for i in 0..updates.len() as usize {
            new_model.aggregate(&updates[i]);
        }

        let new_version = new_model.version;
        let new_version_hash = Address::from_key(SchemaUtils::pubkey_from_version(new_version));
        println!("Created New Model: {:?}", new_model);
        self.public.models.put(&new_version_hash, new_model);
        self.public.latest_version_addr.set(new_version_hash);
    }
}

/// Schema Helpers
#[derive(Debug)]
pub struct SchemaUtils {}

impl SchemaUtils {
    /// Transform version number into public key
    pub fn pubkey_from_version(version: u32) -> PublicKey {
        let mut byte_array: [u8; 32] = [0 as u8; 32];
        let _2b = version.to_be_bytes();
        for i in 0..4 as usize {
            byte_array[i] = _2b[i];
        }

        return PublicKey::new(byte_array);
    }
}
