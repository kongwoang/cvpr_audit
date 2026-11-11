# v3a-specific changed-episode attribution

Earlier traces stored the last (joint-adapt) refinement for all-object/query policies. This audit replays the frozen v3a refiner on the actual stored final raw scenes and original current images, on CPU only, using the same support and features. Every audited action must match the stored GPU v3a action. No acquisition, histories, checkpoint, thresholds or aggregate score changes. All changed episodes vs round1, including lost episodes, are included. A matched decisive object that brakes is evidence, not an identity or safety guarantee.
