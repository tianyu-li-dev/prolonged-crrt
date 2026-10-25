"""Column names and source identifiers for landmark CRRT analysis."""
# Core MIMIC identifiers.
SUBJECT_ID = "subject_id"
HADM_ID = "hadm_id"
STAY_ID = "stay_id"

# Common time columns.
START_TIME = "starttime"
END_TIME = "endtime"
DOD = "dod"
ICU_INTIME = "icu_intime"
ADMISSION_AGE = "admission_age"

# CRRT episode columns.
EPISODE_ID = "episode_id"
EPISODE_START = "episode_start"
EPISODE_END = "episode_end"
GAP_FROM_PREVIOUS_EPISODE = "gap_from_previous_episode"
N_EPISODES_FOR_STAY = "n_episodes_for_stay"
HAS_MULTIPLE_EPISODES = "has_multiple_episodes"

# Analysis columns.
INTERVENTION_STATUS = "intervention_status"
MORTALITY_TEMPLATE = "mortality_{days}days"
