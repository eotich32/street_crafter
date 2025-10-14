from bokeh.io import output_notebook

from tutorials.utils.tutorial_utils import visualize_nuplan_scenarios, setup_notebook

import os
NUPLAN_DATA_ROOT = 'E:/Downloads/nuPlan/dataset'
NUPLAN_MAPS_ROOT = 'E:/Downloads/nuPlan/dataset/maps'
NUPLAN_DB_FILES = 'E:/Downloads/nuPlan/dataset/nuplan-v1.1/splits/mini'
NUPLAN_MAP_VERSION = 'nuplan-maps-v1.0'

visualize_nuplan_scenarios(
    data_root=NUPLAN_DATA_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_root=NUPLAN_MAPS_ROOT,
    map_version=NUPLAN_MAP_VERSION,
    bokeh_port=8899  # This controls the port bokeh uses when generating the visualization -- if you are running
                     # the notebook on a remote instance, you'll need to make sure to port-forward it.
)