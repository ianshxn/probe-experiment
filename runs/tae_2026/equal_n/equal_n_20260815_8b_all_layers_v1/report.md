# All-layer equal-N and target-format exposure trajectory

Run ID: `equal_n_20260815_8b_all_layers_v1`  
Git commit: `4e64e93729026ce39c85e13800e1a40467e4f644`  
Result SHA256: `f064b230f7076562bf68033d03129d890020ef357cdb0b70b0dd143983d711ed`  
Model: Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`; 32 layers; BF16 inference / float32 stored; final prompt token; zero-based block output before final norm  
Input SHA256: `ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022`

All layers use the fixed assignment bank `[1729, 2718, 31415, 4242, 8675309]`, structural row checks, held-out purpose families, and separate AUC/balanced accuracy. Same-format and cross-format reference regimes are included at every layer.

## Numeric inspection

Using the matching all-layer geometry trajectory:

- interaction ratio vs equal-N mixed-format AUC correlation: `-0.1605`;
- interaction ratio vs benchmark→casual AUC correlation: `-0.3818`;
- interaction ratio vs casual→benchmark AUC correlation: `-0.3474`;
- equal-N mixed AUC range: `0.5505–0.9237`, with the trajectory maximum `0.9237` at layer 8;
- layer 24 equal-N mixed / benchmark→casual / casual→benchmark AUC: `0.6684 / 0.5544 / 0.5656`.

The interaction ratio tracks one-format cross-format transfer more strongly than equal-N mixed-format AUC, which is expected because equal-N mixed-format training has access to both formats. This distinguishes target-format exposure from the interaction explanation rather than collapsing them.

The full machine-readable per-layer result includes AUC, balanced accuracy, paired metrics, same-format references, per-family metrics, fixed seed assignments, and conditional family bootstrap intervals. Layers remain repeated measurements, not independent replicates.
