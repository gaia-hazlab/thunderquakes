"""CNN classifier, uncertainty estimation, and training (WS4).

Model A: seismic-only 3-C CNN — the deployable workhorse (all stations, all epochs).
Model B: seismic + infrasound dual-branch — trained/evaluated on co-located
         (TA/N4/AK-TA BDF) subset; ablation-proves the seismo-acoustic delay and
         produces high-confidence auto-labels to clean Model A's training set.
Infrasound is an optional second branch (Model A == Model B with it masked).
Uncertainty: MC-dropout + a small deep ensemble, with calibration (ECE).
"""
