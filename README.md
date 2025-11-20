# MAPS (Multiresolution Adaptive Parametrization of Surfaces)
*Authors*:  
&nbsp;&nbsp;&nbsp;&nbsp;Abdelmounaim Salouani\
&nbsp;&nbsp;&nbsp;&nbsp;Marwane Karaoui\
&nbsp;&nbsp;&nbsp;&nbsp;Yasmina Abou-el-abbas\
&nbsp;&nbsp;&nbsp;&nbsp;Hedi Zahaf


## Paper Analysis

## Code Implementation
run this for rabbit :
# generate a progressive file for bunny (writes obja/example/bunny_maps.obja)
python3 -c "from chunk_7_10_complete_pipeline import MAPSProgressiveEncoder; MAPSProgressiveEncoder(target_base_size=80).process_obj_to_obja('obja/example/bunny.obj', 'obja/example/bunny_maps.obja')"