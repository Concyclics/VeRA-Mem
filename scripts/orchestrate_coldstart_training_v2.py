"""Resume the retained controls and corrected Wikipedia branches after repair.

The caller must first finish repair_coldstart_overlap.py on the recorded cache.
Original v1 warm A/C evidence is preserved and explicitly excluded from analysis.
"""
import orchestrate_coldstart_training as controller

if __name__=='__main__':
    controller.STATE=controller.ROOT.parent/'plans/coldstart_training_controller_v2_20261006.json'
    controller.GROUPS=[
        [('coldstart_warm_a_v2_20261006','0'),('coldstart_warm_c_v2_20261006','3'),
         ('coldstart_warm_b_20261006',None),('coldstart_eval_reused_dev_v2_20261006',None)],
        [('coldstart_dictionary_a_20261006','0'),('coldstart_dictionary_b_20261006','1')],
    ]
    controller.main()
