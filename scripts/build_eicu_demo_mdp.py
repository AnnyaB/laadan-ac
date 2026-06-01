
# build_eicu_demo_mdp.py


# This script converts the official eICU Collaborative Research Database Demo
# into a small tabular Markov Decision Process (MDP) that follows the same file
# layout as the ICU-Sepsis benchmark used in this experiment.


#  note:
# This script creates a small real-data cross-source portability task. It is not
# intended to replace full external clinical validation. The aim is to test
# whether the BC, CQL-inspired, VOAC, and LAADAN-AC pipeline can be reused on a
# second ICU trajectory source with the same evaluation interface.

# argparse is used to read command-line options.
import argparse

# json is used to save transparent metadata for reproducibility.
import json

# os is used to create folders and build file paths.
import os

# re is used for simple keyword-based treatment-action grouping.
import re

# sqlite3 is used to read the downloaded eICU demo SQLite database.
import sqlite3

# numpy is used for numerical arrays and probability matrices.
import numpy as np

# pandas is used for table loading, cleaning, grouping and CSV writing.
import pandas as pd

# scikit-learn is used only for standardisation and clustering of hourly states.
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class CSVTemplateWriter:
    """
    Writes MDP files in a format compatible with the existing project.
    """

    def __init__(self, template_dir):
        """
        Store the template directory.

        If the template directory is missing, the writer falls back to a simple
        dense CSV format that is easy for the benchmark loader to parse.
        """
        self.template_dir = template_dir

    def template_path(self, filename):
        """
        Build the full path to a template file.
        """
        if self.template_dir is None:
            return None
        return os.path.join(self.template_dir, filename)

    def file_exists(self, filename):
        """
        Check whether a matching template file exists.
        """
        path = self.template_path(filename)
        return path is not None and os.path.exists(path)

    def first_non_empty_line(self, path):
        """
        Return the first non-empty line in a text file.
        """
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    return stripped
        return ""

    def has_header(self, path):
        """
        Estimate whether a CSV file has a header by checking the first line.
        """
        first = self.first_non_empty_line(path)
        return any(character.isalpha() for character in first)

    def write_vector(self, path, values, template_filename=None, column_name="value"):
        """
        Write a one-dimensional numeric vector with no CSV header.

        The existing benchmark loader reads this file with numpy.loadtxt, so
        rewardFunction.csv and initialStateDistribution.csv must contain numbers
        only. Header names such as "reward" or "probability" would make loading
        fail.
        """
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        frame = pd.DataFrame(values)
        frame.to_csv(path, index=False, header=False)

    def write_matrix(self, path, matrix, template_filename=None):
        """
        Write a two-dimensional numeric matrix with no CSV header.

        The existing benchmark loader reads expertPolicy.csv and
        stateClusterCenters.csv with numpy.loadtxt, so these files must contain
        numbers only.
        """
        matrix = np.asarray(matrix, dtype=float)
        frame = pd.DataFrame(matrix)
        frame.to_csv(path, index=False, header=False)

    def transition_format(self):
        """
        Detect the transitionFunction.csv format used by the template.

        Supported formats:
        - sparse: columns contain state/action/next_state/probability values.
        - dense: rows are state-action pairs and columns are next-state probabilities.
        """
        if not self.file_exists("transitionFunction.csv"):
            return "dense"

        path = self.template_path("transitionFunction.csv")
        first = self.first_non_empty_line(path)
        parts = [part.strip() for part in first.split(",")]

        if any(part.lower() in ["state", "s", "action", "a", "next_state", "probability", "prob"] for part in parts):
            return "sparse"

        if len(parts) == 4:
            return "sparse"

        return "dense"

    def write_transition(self, path, transition):
        """
        Write transition probabilities in the detected format.

        transition has shape:
        number of states × number of actions × number of states
        """
        transition = np.asarray(transition, dtype=float)
        num_states, num_actions, _ = transition.shape
        fmt = self.transition_format()

        if fmt == "sparse":
            rows = []
            for state in range(num_states):
                for action in range(num_actions):
                    next_probs = transition[state, action]
                    next_states = np.flatnonzero(next_probs > 0.0)
                    for next_state in next_states:
                        rows.append({
                            "state": state,
                            "action": action,
                            "next_state": int(next_state),
                            "probability": float(next_probs[next_state]),
                        })

            frame = pd.DataFrame(rows, columns=["state", "action", "next_state", "probability"])
            frame.to_csv(path, index=False, header=False)
            return

        dense = transition.reshape(num_states * num_actions, num_states)
        frame = pd.DataFrame(dense)
        frame.to_csv(path, index=False, header=False)

    def admissible_format(self):
        """
        Detect how admissibleActions.txt is written in the template.

        Supported formats:
        - colon_list: state: a,b,c
        - csv_pairs: state,action pairs, one per line
        - csv_list: state,action1,action2,...
        - space_list: state action1 action2 ...
        """
        if not self.file_exists("admissibleActions.txt"):
            return "space_list"

        path = self.template_path("admissibleActions.txt")
        first = self.first_non_empty_line(path)

        if ":" in first:
            return "colon_list"

        comma_parts = [part.strip() for part in first.split(",")]
        if "," in first and len(comma_parts) == 2:
            return "csv_pairs"

        if "," in first:
            return "csv_list"

        return "space_list"

    def write_admissible_actions(self, path, admissible_mask):
        """
        Write admissible actions in the detected template style.
        """
        admissible_mask = np.asarray(admissible_mask, dtype=bool)
        num_states, _ = admissible_mask.shape
        fmt = self.admissible_format()

        with open(path, "w", encoding="utf-8") as handle:
            if fmt == "csv_pairs":
                for state in range(num_states):
                    actions = np.flatnonzero(admissible_mask[state])
                    for action in actions:
                        handle.write(str(state) + "," + str(int(action)) + "\n")
                return

            for state in range(num_states):
                actions = [str(int(action)) for action in np.flatnonzero(admissible_mask[state])]

                if fmt == "colon_list":
                    handle.write(str(state) + ": " + ",".join(actions) + "\n")
                elif fmt == "csv_list":
                    handle.write(",".join([str(state)] + actions) + "\n")
                else:
                    handle.write(" ".join([str(state)] + actions) + "\n")


class EICUDemoMDPBuilder:
    """
    Converts eICU-CRD Demo clinical tables into a compact tabular MDP.

    The design deliberately favours transparency over complexity:
    - hourly ICU states are created from vital signs, selected labs and patient metadata;
    - states are discretised using MiniBatchKMeans;
    - clinician treatment records are mapped to a small set of treatment-intensity actions;
    - expert policy and admissibility mask are estimated from empirical state-action support;
    - terminal reward is based on hospital/unit discharge survival status.
    """

    def __init__(self, args):
        """
        Store command-line arguments and initialise derived paths.
        """
        self.args = args
        self.db_path = args.db_path
        self.output_dir = args.output_dir
        self.extras_dir = os.path.join(self.output_dir, "extras")
        self.writer = CSVTemplateWriter(args.template_dir)

        # The first eICU adapter used five broad treatment categories. For the
        # cross-domain portability experiment we default to an ICU-Sepsis-style
        # 25-action grid, created from two clinically interpretable treatment axes.
        # This gives the safe policies more than one supported action in many
        # states, instead of collapsing all masked methods to the same action.
        self.action_names = self.build_action_names()

    def build_action_names(self):
        """
        Build the action-name dictionary used in the generated MDP.

        The 25-action mode follows the same spirit as the ICU-Sepsis benchmark:
        Each action is a combination of two discrete treatment dimensions.
        The dimensions are not copied from ICU-Sepsis, because the eICU demo exposes
        different treatment tables. Instead, they are estimated from real eICU
        medication, infusion, treatment, and respiratory-care records.
        """
        if int(self.args.action_grid) == 5:
            return {
                0: "no_recorded_acute_treatment",
                1: "fluid_or_general_infusion",
                2: "antibiotic_or_antiinfective",
                3: "respiratory_support",
                4: "vasopressor_or_cardiovascular_support",
            }

        hemodynamic_levels = [
            "hemo_0_none",
            "hemo_1_fluid_low",
            "hemo_2_fluid_high",
            "hemo_3_vasopressor_or_cardiac",
            "hemo_4_combined_hemodynamic_support",
        ]
        support_levels = [
            "support_0_none",
            "support_1_antibiotic",
            "support_2_respiratory",
            "support_3_antibiotic_and_respiratory",
            "support_4_high_combined_support",
        ]

        names = {}
        index = 0
        for hemo_name in hemodynamic_levels:
            for support_name in support_levels:
                names[index] = hemo_name + "__" + support_name
                index += 1
        return names

    def combine_action_axes(self, fluid_count, antibiotic_count, respiratory_count, vasopressor_count):
        """
        Converting hourly treatment counts into either 5 or 25 discrete actions.

        In 25-action mode, the first axis describes haemodynamic support and the
        second axis describes anti-infective/respiratory support. The action is:

            action = 5 * haemodynamic_bin + support_bin

        This is a real-data discretisation: all counts come from observed eICU
        clinical records. It is still an approximation and should be described as
        a portability MDP.
        """
        fluid_count = int(fluid_count)
        antibiotic_count = int(antibiotic_count)
        respiratory_count = int(respiratory_count)
        vasopressor_count = int(vasopressor_count)

        if int(self.args.action_grid) == 5:
            if vasopressor_count > 0:
                return 4
            if respiratory_count > 0:
                return 3
            if antibiotic_count > 0:
                return 2
            if fluid_count > 0:
                return 1
            return 0

        if vasopressor_count > 0 and fluid_count > 0:
            hemo_bin = 4
        elif vasopressor_count > 0:
            hemo_bin = 3
        elif fluid_count >= 2:
            hemo_bin = 2
        elif fluid_count == 1:
            hemo_bin = 1
        else:
            hemo_bin = 0

        if antibiotic_count > 0 and respiratory_count > 0:
            support_bin = 4 if (antibiotic_count + respiratory_count) >= 3 else 3
        elif respiratory_count > 0:
            support_bin = 2
        elif antibiotic_count > 0:
            support_bin = 1
        else:
            support_bin = 0

        return hemo_bin * 5 + support_bin

    def run(self):
        """
        Run the complete build pipeline.
        """
        self.prepare_output_folders()

        print("[STEP 1] Loading patient outcomes...", flush=True)
        patients = self.load_patients()

        print("[STEP 2] Building hourly state table from vitals, labs and metadata...", flush=True)
        hourly_states = self.build_hourly_state_table(patients)

        print("[STEP 3] Building hourly treatment-action labels...", flush=True)
        hourly_actions = self.build_hourly_action_table(hourly_states)

        print("[STEP 4] Merging states and actions into patient-hour records...", flush=True)
        records = self.merge_states_and_actions(hourly_states, hourly_actions, patients)

        print("[STEP 5] Clustering real ICU patient-hours into discrete MDP states...", flush=True)
        records, centres = self.cluster_states(records)

        print("[STEP 6] Estimating transitions, rewards, expert policy and admissibility mask...", flush=True)
        mdp = self.build_mdp_matrices(records, centres)

        print("[STEP 7] Writing ICU-Sepsis-style MDP files...", flush=True)
        self.write_outputs(mdp)

        print("[DONE] eICU demo MDP saved to:", self.output_dir, flush=True)
        print("[DONE] Number of non-terminal states:", mdp["num_nonterminal_states"], flush=True)
        print("[DONE] Number of total states:", mdp["num_states"], flush=True)
        print("[DONE] Number of actions:", mdp["num_actions"], flush=True)
        print("[DONE] Number of ICU stays used:", mdp["num_episodes"], flush=True)
        print("[DONE] Number of observed transitions:", mdp["num_observed_transitions"], flush=True)

    def prepare_output_folders(self):
        """
        Create output folders if they do not exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.extras_dir, exist_ok=True)

    def connect(self):
        """
        Open a SQLite connection to the eICU demo database.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Could not find SQLite database: " + self.db_path)
        return sqlite3.connect(self.db_path)

    def table_columns(self, connection, table_name):
        """
        Return all column names for a SQLite table.
        """
        rows = connection.execute("PRAGMA table_info(" + table_name + ")").fetchall()
        return [row[1] for row in rows]

    def read_table(self, connection, table_name, preferred_columns=None):
        """
        Read a table using only the requested columns that actually exist.

        This makes the script more robust to small schema differences between
        eICU demo releases.
        """
        columns = self.table_columns(connection, table_name)

        if preferred_columns is None:
            selected = columns
        else:
            selected = [column for column in preferred_columns if column in columns]

        if len(selected) == 0:
            raise ValueError("None of the requested columns exist in table: " + table_name)

        sql = "SELECT " + ", ".join(selected) + " FROM " + table_name
        return pd.read_sql_query(sql, connection)

    def load_patients(self):
        """
        Load patient-level information and create a binary terminal outcome.

        outcome = 1 means survival/alive discharge.
        outcome = 0 means expired/death.
        """
        with self.connect() as connection:
            patient = self.read_table(
                connection,
                "patient",
                [
                    "patientunitstayid",
                    "gender",
                    "age",
                    "admissionheight",
                    "admissionweight",
                    "unitdischargeoffset",
                    "hospitaldischargeoffset",
                    "unitdischargestatus",
                    "hospitaldischargestatus",
                ],
            )

        patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce")
        patient = patient.dropna(subset=["patientunitstayid"])
        patient["patientunitstayid"] = patient["patientunitstayid"].astype(int)

        if "age" in patient.columns:
            patient["age_numeric"] = patient["age"].apply(self.parse_age)
        else:
            patient["age_numeric"] = np.nan

        if "gender" in patient.columns:
            patient["gender_numeric"] = patient["gender"].astype(str).str.lower().map({
                "male": 1.0,
                "female": 0.0,
            })
        else:
            patient["gender_numeric"] = np.nan

        patient["terminal_outcome"] = patient.apply(self.infer_outcome, axis=1)
        patient = patient.dropna(subset=["terminal_outcome"])
        patient["terminal_outcome"] = patient["terminal_outcome"].astype(int)

        return patient

    def parse_age(self, value):
        """
        Convert eICU age strings into numeric values.
        """
        if pd.isna(value):
            return np.nan

        text = str(value).strip().replace(">", "")
        try:
            return float(text)
        except ValueError:
            return np.nan

    def infer_outcome(self, row):
        """
        Infer survival status from hospital or unit discharge status.
        """
        fields = []
        for column in ["hospitaldischargestatus", "unitdischargestatus"]:
            if column in row and not pd.isna(row[column]):
                fields.append(str(row[column]).lower())

        joined = " ".join(fields)

        if "expired" in joined or "death" in joined or "dead" in joined:
            return 0

        if "alive" in joined or "survive" in joined or "home" in joined:
            return 1

        return np.nan

    def build_hourly_state_table(self, patients):
        """
        Build one state-feature row per patient-hour.

        Vitals come from vitalperiodic. Selected labs are forward-filled within
        each ICU stay, then merged onto the nearest patient-hour.
        """
        with self.connect() as connection:
            vitals = self.read_table(
                connection,
                "vitalperiodic",
                [
                    "patientunitstayid",
                    "observationoffset",
                    "temperature",
                    "sao2",
                    "heartrate",
                    "respiration",
                    "systemicsystolic",
                    "systemicdiastolic",
                    "systemicmean",
                ],
            )

            labs = self.read_table(
                connection,
                "lab",
                [
                    "patientunitstayid",
                    "labresultoffset",
                    "labname",
                    "labresult",
                ],
            )

        vitals["patientunitstayid"] = pd.to_numeric(vitals["patientunitstayid"], errors="coerce")
        vitals["observationoffset"] = pd.to_numeric(vitals["observationoffset"], errors="coerce")
        vitals = vitals.dropna(subset=["patientunitstayid", "observationoffset"])
        vitals["patientunitstayid"] = vitals["patientunitstayid"].astype(int)

        vitals = vitals[vitals["observationoffset"] >= 0]
        vitals = vitals[vitals["observationoffset"] <= int(self.args.max_hours * 60)]
        vitals["hour"] = (vitals["observationoffset"] // 60).astype(int)

        vital_features = [
            column for column in [
                "temperature",
                "sao2",
                "heartrate",
                "respiration",
                "systemicsystolic",
                "systemicdiastolic",
                "systemicmean",
            ]
            if column in vitals.columns
        ]

        for column in vital_features:
            vitals[column] = pd.to_numeric(vitals[column], errors="coerce")

        hourly = (
            vitals
            .groupby(["patientunitstayid", "hour"], as_index=False)[vital_features]
            .mean()
        )

        lab_features = self.build_hourly_labs(labs)
        if lab_features is not None and len(lab_features) > 0:
            hourly = hourly.merge(lab_features, on=["patientunitstayid", "hour"], how="left")

        metadata_columns = [
            column for column in [
                "patientunitstayid",
                "age_numeric",
                "gender_numeric",
                "admissionheight",
                "admissionweight",
                "terminal_outcome",
            ]
            if column in patients.columns
        ]
        metadata = patients[metadata_columns].copy()

        for column in ["admissionheight", "admissionweight"]:
            if column in metadata.columns:
                metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

        hourly = hourly.merge(metadata, on="patientunitstayid", how="inner")

        feature_columns = [
            column for column in hourly.columns
            if column not in ["patientunitstayid", "hour", "terminal_outcome"]
        ]

        hourly = hourly.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

        for column in feature_columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
            hourly[column] = hourly.groupby("patientunitstayid")[column].ffill()
            hourly[column] = hourly.groupby("patientunitstayid")[column].bfill()

        hourly = hourly.dropna(subset=vital_features, how="all")

        for column in feature_columns:
            if hourly[column].isna().any():
                hourly[column] = hourly[column].fillna(hourly[column].median())

        counts = hourly.groupby("patientunitstayid").size()
        valid_stays = counts[counts >= int(self.args.min_hours_per_stay)].index
        hourly = hourly[hourly["patientunitstayid"].isin(valid_stays)].reset_index(drop=True)

        if len(hourly) == 0:
            raise RuntimeError("No usable hourly state records were created.")

        return hourly

    def build_hourly_labs(self, labs):
        """
        Convert selected lab measurements into hourly wide features.
        """
        if labs is None or len(labs) == 0:
            return None

        required = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
        for column in required:
            if column not in labs.columns:
                return None

        labs = labs.copy()
        labs["patientunitstayid"] = pd.to_numeric(labs["patientunitstayid"], errors="coerce")
        labs["labresultoffset"] = pd.to_numeric(labs["labresultoffset"], errors="coerce")
        labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
        labs = labs.dropna(subset=["patientunitstayid", "labresultoffset", "labname", "labresult"])
        labs = labs[labs["labresultoffset"] >= 0]
        labs = labs[labs["labresultoffset"] <= int(self.args.max_hours * 60)]

        labs["patientunitstayid"] = labs["patientunitstayid"].astype(int)
        labs["hour"] = (labs["labresultoffset"] // 60).astype(int)
        labs["labname_clean"] = labs["labname"].astype(str).str.lower().str.strip()

        selected_patterns = {
            "lab_glucose": ["glucose"],
            "lab_creatinine": ["creatinine"],
            "lab_bun": ["bun"],
            "lab_sodium": ["sodium"],
            "lab_potassium": ["potassium"],
            "lab_bicarbonate": ["bicarbonate", "hco3"],
            "lab_chloride": ["chloride"],
            "lab_hemoglobin": ["hgb", "hemoglobin"],
            "lab_wbc": ["wbc"],
            "lab_platelets": ["platelets"],
            "lab_lactate": ["lactate"],
        }

        frames = []
        for output_name, patterns in selected_patterns.items():
            mask = np.zeros(len(labs), dtype=bool)
            for pattern in patterns:
                mask = mask | labs["labname_clean"].str.contains(pattern, na=False, regex=False).to_numpy()

            subset = labs[mask]
            if len(subset) == 0:
                continue

            grouped = (
                subset
                .groupby(["patientunitstayid", "hour"], as_index=False)["labresult"]
                .mean()
                .rename(columns={"labresult": output_name})
            )
            frames.append(grouped)

        if len(frames) == 0:
            return None

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["patientunitstayid", "hour"], how="outer")

        merged = merged.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)
        lab_columns = [column for column in merged.columns if column not in ["patientunitstayid", "hour"]]

        for column in lab_columns:
            merged[column] = merged.groupby("patientunitstayid")[column].ffill()

        return merged

    def build_hourly_action_table(self, hourly_states):
        """
        Assign one treatment-action label to each patient-hour.

        The improved default uses a 25-action ICU-Sepsis-style grid rather than
        only five broad categories.

        The 25-action grid keeps the task real-data-based while giving the agent
        more supported choices:
        - haemodynamic axis: none, low fluid, high fluid, vasopressor/cardiac, combined;
        - support axis: none, antibiotic, respiratory, antibiotic+respiratory, high combined.
        """
        base = hourly_states[["patientunitstayid", "hour"]].copy()

        action_events = self.load_action_events()

        if len(action_events) == 0:
            base["fluid_count"] = 0
            base["antibiotic_count"] = 0
            base["respiratory_count"] = 0
            base["vasopressor_count"] = 0
            base["action"] = 0
            return base[["patientunitstayid", "hour", "action"]]

        action_events = action_events[action_events["hour"] >= 0]
        action_events = action_events[action_events["hour"] <= int(self.args.max_hours)]

        action_events["fluid_event"] = (action_events["action"] == 1).astype(int)
        action_events["antibiotic_event"] = (action_events["action"] == 2).astype(int)
        action_events["respiratory_event"] = (action_events["action"] == 3).astype(int)
        action_events["vasopressor_event"] = (action_events["action"] == 4).astype(int)

        hourly_counts = (
            action_events
            .groupby(["patientunitstayid", "hour"], as_index=False)[
                ["fluid_event", "antibiotic_event", "respiratory_event", "vasopressor_event"]
            ]
            .sum()
            .rename(columns={
                "fluid_event": "fluid_count",
                "antibiotic_event": "antibiotic_count",
                "respiratory_event": "respiratory_count",
                "vasopressor_event": "vasopressor_count",
            })
        )

        merged = base.merge(hourly_counts, on=["patientunitstayid", "hour"], how="left")

        for column in ["fluid_count", "antibiotic_count", "respiratory_count", "vasopressor_count"]:
            merged[column] = merged[column].fillna(0).astype(int)

        merged["action"] = merged.apply(
            lambda row: self.combine_action_axes(
                row["fluid_count"],
                row["antibiotic_count"],
                row["respiratory_count"],
                row["vasopressor_count"],
            ),
            axis=1,
        )

        merged["action"] = merged["action"].astype(int)
        return merged[["patientunitstayid", "hour", "action"]]


    def load_action_events(self):
        """
        Load medication, infusion and treatment records and map them to actions.
        """
        frames = []

        with self.connect() as connection:
            table_names = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                connection,
            )["name"].tolist()

            if "medication" in table_names:
                medication = self.read_table(
                    connection,
                    "medication",
                    ["patientunitstayid", "drugorderoffset", "drugstartoffset", "drugname"],
                )
                frames.append(self.events_from_medication(medication))

            if "infusiondrug" in table_names:
                infusion = self.read_table(
                    connection,
                    "infusiondrug",
                    ["patientunitstayid", "infusionoffset", "drugname"],
                )
                frames.append(self.events_from_infusion(infusion))

            if "treatment" in table_names:
                treatment = self.read_table(
                    connection,
                    "treatment",
                    ["patientunitstayid", "treatmentoffset", "treatmentstring"],
                )
                frames.append(self.events_from_treatment(treatment))

            if "respiratorycare" in table_names:
                respiratory = self.read_respiratorycare_table(connection)
                frames.append(self.events_from_respiratory(respiratory))

        frames = [frame for frame in frames if frame is not None and len(frame) > 0]
        if len(frames) == 0:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "action"])

        events = pd.concat(frames, ignore_index=True)
        events = events.dropna(subset=["patientunitstayid", "hour", "action"])
        events["patientunitstayid"] = events["patientunitstayid"].astype(int)
        events["hour"] = events["hour"].astype(int)
        events["action"] = events["action"].astype(int)

        return events

    def events_from_medication(self, medication):
        """
        Convert medication rows to action events.
        """
        if medication is None or len(medication) == 0 or "drugname" not in medication.columns:
            return None

        medication = medication.copy()
        offset_column = "drugstartoffset" if "drugstartoffset" in medication.columns else "drugorderoffset"
        if offset_column not in medication.columns:
            return None

        medication["offset"] = pd.to_numeric(medication[offset_column], errors="coerce")
        medication["hour"] = (medication["offset"] // 60).astype("Int64")
        medication["text"] = medication["drugname"].astype(str).str.lower()
        medication["action"] = medication["text"].apply(self.classify_action_text)

        return medication[["patientunitstayid", "hour", "action"]]

    def events_from_infusion(self, infusion):
        """
        Convert infusion rows to action events.
        """
        if infusion is None or len(infusion) == 0 or "drugname" not in infusion.columns:
            return None

        infusion = infusion.copy()
        if "infusionoffset" not in infusion.columns:
            return None

        infusion["offset"] = pd.to_numeric(infusion["infusionoffset"], errors="coerce")
        infusion["hour"] = (infusion["offset"] // 60).astype("Int64")
        infusion["text"] = infusion["drugname"].astype(str).str.lower()
        infusion["action"] = infusion["text"].apply(self.classify_action_text)

        return infusion[["patientunitstayid", "hour", "action"]]

    def events_from_treatment(self, treatment):
        """
        Convert treatment rows to action events.
        """
        if treatment is None or len(treatment) == 0 or "treatmentstring" not in treatment.columns:
            return None

        treatment = treatment.copy()
        if "treatmentoffset" not in treatment.columns:
            return None

        treatment["offset"] = pd.to_numeric(treatment["treatmentoffset"], errors="coerce")
        treatment["hour"] = (treatment["offset"] // 60).astype("Int64")
        treatment["text"] = treatment["treatmentstring"].astype(str).str.lower()
        treatment["action"] = treatment["text"].apply(self.classify_action_text)

        return treatment[["patientunitstayid", "hour", "action"]]

    def read_respiratorycare_table(self, connection):
        """
        Read respiratory-care rows only when usable columns exist.

        The eICU demo schema can differ from the full eICU schema. Some
        releases do not expose respcarestatusoffset and airwaytype. In that
        case, this function skips respiratorycare safely instead of stopping
        the whole MDP build.
        """
        columns = self.table_columns(connection, "respiratorycare")

        if "patientunitstayid" not in columns:
            return pd.DataFrame(columns=["patientunitstayid", "hour", "action"])

        offset_candidates = [
            "respcarestatusoffset",
            "respchartoffset",
            "respchartentryoffset",
            "treatmentoffset",
            "observationoffset",
        ]

        text_candidates = [
            "airwaytype",
            "ventstartoffset",
            "ventendoffset",
            "respcarestatus",
            "respchartvaluelabel",
            "respchartvalue",
        ]

        offset_column = None
        for candidate in offset_candidates:
            if candidate in columns:
                offset_column = candidate
                break

        if offset_column is None:
            print(
                "[WARN] respiratorycare table found, but no recognised offset column was available. Skipping respiratorycare actions.",
                flush=True,
            )
            return pd.DataFrame(columns=["patientunitstayid", "hour", "action"])

        selected = ["patientunitstayid", offset_column]
        for candidate in text_candidates:
            if candidate in columns and candidate not in selected:
                selected.append(candidate)

        table = self.read_table(connection, "respiratorycare", selected)
        table = table.rename(columns={offset_column: "respiratory_offset"})

        text_columns = [column for column in table.columns if column not in ["patientunitstayid", "respiratory_offset"]]
        if len(text_columns) == 0:
            table["respiratory_text"] = "respiratory support"
        else:
            table["respiratory_text"] = ""
            for column in text_columns:
                table["respiratory_text"] = table["respiratory_text"] + " " + table[column].astype(str)

        return table[["patientunitstayid", "respiratory_offset", "respiratory_text"]]

    def events_from_respiratory(self, respiratory):
        """
        Convert respiratory-care rows to action events.
        """
        if respiratory is None or len(respiratory) == 0:
            return None

        respiratory = respiratory.copy()

        if "respiratory_offset" not in respiratory.columns:
            return None

        respiratory["offset"] = pd.to_numeric(respiratory["respiratory_offset"], errors="coerce")
        respiratory = respiratory.dropna(subset=["patientunitstayid", "offset"])
        if len(respiratory) == 0:
            return None

        respiratory["hour"] = (respiratory["offset"] // 60).astype("Int64")

        if "respiratory_text" in respiratory.columns:
            respiratory["text"] = respiratory["respiratory_text"].astype(str).str.lower()
            respiratory["action"] = respiratory["text"].apply(self.classify_action_text)
            respiratory.loc[respiratory["action"] == 0, "action"] = 3
        else:
            respiratory["action"] = 3

        return respiratory[["patientunitstayid", "hour", "action"]]

    def classify_action_text(self, text):
        """
        Map a clinical text field to one of the five discrete action categories.
        """
        if pd.isna(text):
            return 0

        text = str(text).lower()

        vasopressor_pattern = (
            r"norepinephrine|levophed|epinephrine|dopamine|dobutamine|"
            r"phenylephrine|neo-synephrine|neosynephrine|vasopressin|milrinone"
        )
        respiratory_pattern = (
            r"ventilator|ventilation|intubat|extubat|peep|fio2|oxygen|"
            r"respiratory|airway|cpap|bipap"
        )
        antibiotic_pattern = (
            r"antibiotic|vancomycin|cef|cillin|penem|floxacin|cycline|"
            r"azithro|metronidazole|zosyn|piperacillin|tazobactam|gentamicin"
        )
        fluid_pattern = (
            r"saline|ringer|lactated|fluid|albumin|dextrose|sodium chloride|"
            r"normal saline|ns "
        )

        if re.search(vasopressor_pattern, text):
            return 4
        if re.search(respiratory_pattern, text):
            return 3
        if re.search(antibiotic_pattern, text):
            return 2
        if re.search(fluid_pattern, text):
            return 1

        return 0

    def merge_states_and_actions(self, hourly_states, hourly_actions, patients):
        """
        Merge hourly state features, action labels and terminal outcomes.
        """
        records = hourly_states.merge(
            hourly_actions,
            on=["patientunitstayid", "hour"],
            how="left",
        )

        records["action"] = records["action"].fillna(0).astype(int)

        outcome = patients[["patientunitstayid", "terminal_outcome"]].copy()
        records = records.drop(columns=["terminal_outcome"], errors="ignore")
        records = records.merge(outcome, on="patientunitstayid", how="inner")

        records = records.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

        return records

    def feature_columns(self, records):
        """
        Return feature columns used for state clustering.
        """
        excluded = {
            "patientunitstayid",
            "hour",
            "terminal_outcome",
            "action",
            "cluster_state",
        }
        return [column for column in records.columns if column not in excluded]

    def cluster_states(self, records):
        """
        Standardise features and cluster patient-hours into discrete states.
        """
        feature_columns = self.feature_columns(records)
        features = records[feature_columns].copy()

        for column in feature_columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
            if features[column].isna().any():
                features[column] = features[column].fillna(features[column].median())

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.to_numpy(dtype=float))

        requested_clusters = int(self.args.num_clusters)
        num_clusters = min(requested_clusters, max(2, len(records) // 5))

        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=int(self.args.seed),
            batch_size=min(1024, max(32, len(records))),
            n_init=10,
        )

        records = records.copy()
        records["cluster_state"] = kmeans.fit_predict(features_scaled)

        centres_scaled = kmeans.cluster_centers_

        return records, centres_scaled

    def build_mdp_matrices(self, records, centres):
        """
        Estimate MDP matrices from clustered patient-hour records.
        """
        num_nonterminal_states = int(centres.shape[0])
        death_state = num_nonterminal_states
        survival_state = num_nonterminal_states + 1
        num_states = num_nonterminal_states + 2
        num_actions = len(self.action_names)

        transition_counts = np.zeros((num_states, num_actions, num_states), dtype=float)
        state_action_counts = np.zeros((num_states, num_actions), dtype=float)
        initial_counts = np.zeros(num_states, dtype=float)

        num_observed_transitions = 0
        num_episodes = 0

        for patient_id, group in records.groupby("patientunitstayid"):
            group = group.sort_values("hour").reset_index(drop=True)

            if len(group) < int(self.args.min_hours_per_stay):
                continue

            num_episodes += 1
            first_state = int(group.loc[0, "cluster_state"])
            initial_counts[first_state] += 1.0

            for index in range(len(group)):
                state = int(group.loc[index, "cluster_state"])
                action = int(group.loc[index, "action"])
                action = max(0, min(action, num_actions - 1))

                state_action_counts[state, action] += 1.0

                if index < len(group) - 1:
                    current_hour = int(group.loc[index, "hour"])
                    next_hour = int(group.loc[index + 1, "hour"])

                    if next_hour - current_hour > int(self.args.max_gap_hours):
                        continue

                    next_state = int(group.loc[index + 1, "cluster_state"])
                else:
                    outcome = int(group.loc[index, "terminal_outcome"])
                    next_state = survival_state if outcome == 1 else death_state

                transition_counts[state, action, next_state] += 1.0
                num_observed_transitions += 1

        transition = self.normalise_transitions(transition_counts)
        initial = self.normalise_initial_distribution(initial_counts)
        reward = np.zeros(num_states, dtype=float)
        reward[survival_state] = 1.0

        expert_policy = self.build_expert_policy(state_action_counts)
        admissible_mask = self.build_admissible_mask(state_action_counts)

        terminal_centres = np.zeros((2, centres.shape[1]), dtype=float)
        all_centres = np.vstack([centres, terminal_centres])

        return {
            "transition": transition,
            "reward": reward,
            "initial": initial,
            "expert_policy": expert_policy,
            "admissible_mask": admissible_mask,
            "state_centres": all_centres,
            "state_action_counts": state_action_counts,
            "num_nonterminal_states": num_nonterminal_states,
            "num_states": num_states,
            "num_actions": num_actions,
            "death_state": death_state,
            "survival_state": survival_state,
            "num_episodes": num_episodes,
            "num_observed_transitions": num_observed_transitions,
        }

    def normalise_transitions(self, transition_counts):
        """
        Convert transition counts into transition probabilities.

        Missing state-action rows are filled with self-loops so that every row is
        a valid probability distribution.
        """
        transition = transition_counts.copy()
        num_states, num_actions, _ = transition.shape

        for state in range(num_states):
            for action in range(num_actions):
                total = transition[state, action].sum()
                if total > 0:
                    transition[state, action] = transition[state, action] / total
                else:
                    transition[state, action, state] = 1.0

        return transition

    def normalise_initial_distribution(self, initial_counts):
        """
        Convert initial-state counts into a probability distribution.
        """
        total = initial_counts.sum()
        if total <= 0:
            initial_counts[0] = 1.0
            total = 1.0
        return initial_counts / total

    def build_expert_policy(self, state_action_counts):
        """
        Estimate the expert policy from empirical clinician action frequencies.
        """
        counts = state_action_counts.copy()
        num_states, num_actions = counts.shape
        expert = np.zeros((num_states, num_actions), dtype=float)

        for state in range(num_states):
            total = counts[state].sum()
            if total > 0:
                expert[state] = counts[state] / total
            else:
                expert[state] = 1.0 / float(num_actions)

        return expert

    def build_admissible_mask(self, state_action_counts):
        """
        Estimate admissible actions from empirical state-action support.

        This version is less brittle than the first adapter. It marks actions as
        admissible when they meet the minimum support threshold, but it also keeps
        at least a small number of observed top actions per non-terminal state.
        That makes the eICU portability MDP closer to ICU-Sepsis, where the mask
        restricts unsafe actions but does not usually reduce every state to one
        deterministic choice.
        """
        counts = state_action_counts.copy()
        num_states, num_actions = counts.shape
        mask = np.zeros((num_states, num_actions), dtype=bool)

        min_count = int(self.args.min_action_support)
        min_choices = max(1, int(self.args.min_admissible_actions))

        for state in range(num_states):
            row = counts[state]
            observed_actions = np.flatnonzero(row > 0)

            if observed_actions.size == 0:
                mask[state, :] = True
                continue

            supported = row >= min_count
            mask[state] = supported

            if int(mask[state].sum()) < min_choices:
                ordered = observed_actions[np.argsort(row[observed_actions])[::-1]]
                for action in ordered[:min_choices]:
                    mask[state, int(action)] = True

            if not mask[state].any():
                mask[state, int(np.argmax(row))] = True

        return mask


    def write_outputs(self, mdp):
        """
        Write all MDP files required by the existing training pipeline.
        """
        transition_path = os.path.join(self.output_dir, "transitionFunction.csv")
        reward_path = os.path.join(self.output_dir, "rewardFunction.csv")
        initial_path = os.path.join(self.output_dir, "initialStateDistribution.csv")
        expert_path = os.path.join(self.output_dir, "expertPolicy.csv")
        admissible_path = os.path.join(self.output_dir, "admissibleActions.txt")
        centres_path = os.path.join(self.extras_dir, "stateClusterCenters.csv")

        self.writer.write_transition(transition_path, mdp["transition"])
        self.writer.write_vector(reward_path, mdp["reward"], "rewardFunction.csv", column_name="reward")
        self.writer.write_vector(initial_path, mdp["initial"], "initialStateDistribution.csv", column_name="probability")
        self.writer.write_matrix(expert_path, mdp["expert_policy"], "expertPolicy.csv")
        self.writer.write_admissible_actions(admissible_path, mdp["admissible_mask"])
        self.writer.write_matrix(centres_path, mdp["state_centres"], "stateClusterCenters.csv")

        description = {
            "dataset": "eICU Collaborative Research Database Demo v2.0.1",
            "task_type": "small real-data cross-source portability MDP",
            "source_database": self.db_path,
            "num_states": int(mdp["num_states"]),
            "num_nonterminal_states": int(mdp["num_nonterminal_states"]),
            "num_actions": int(mdp["num_actions"]),
            "feature_dim": int(mdp["state_centres"].shape[1]),
            "death_state": int(mdp["death_state"]),
            "survival_state": int(mdp["survival_state"]),
            "horizon": int(self.args.horizon),
            "max_hours_used": int(self.args.max_hours),
            "num_episodes": int(mdp["num_episodes"]),
            "num_observed_transitions": int(mdp["num_observed_transitions"]),
            "min_action_support": int(self.args.min_action_support),
            "state_representation": "standardised clustered hourly vitals, selected labs and patient metadata",
            "action_definition": self.action_names,
            "reward_definition": "terminal survival/discharge proxy receives reward 1; death and intermediate states receive reward 0",
            "action_grid": int(self.args.action_grid),
            "min_admissible_actions": int(self.args.min_admissible_actions),
            "admissibility_definition": "empirical state-action support threshold plus top observed supported actions per state",
            "important_limitation": "This is a small real-data portability check and not a full external clinical validation experiment.",
        }

        with open(os.path.join(self.extras_dir, "eicu_demo_mdp_description.json"), "w", encoding="utf-8") as handle:
            json.dump(description, handle, indent=2)

        with open(os.path.join(self.extras_dir, "action_names.json"), "w", encoding="utf-8") as handle:
            json.dump(self.action_names, handle, indent=2)

        summary = pd.DataFrame([{
            "num_states": mdp["num_states"],
            "num_nonterminal_states": mdp["num_nonterminal_states"],
            "num_actions": mdp["num_actions"],
            "feature_dim": mdp["state_centres"].shape[1],
            "death_state": mdp["death_state"],
            "survival_state": mdp["survival_state"],
            "num_episodes": mdp["num_episodes"],
            "num_observed_transitions": mdp["num_observed_transitions"],
        }])
        summary.to_csv(os.path.join(self.extras_dir, "build_summary.csv"), index=False)


def parse_args():
    """
    Read command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build a small ICU-Sepsis-style MDP from the eICU-CRD Demo SQLite database."
    )

    parser.add_argument(
        "--db-path",
        default="data/eicu_crd_demo_raw/sqlite/eicu_v2_0_1.sqlite3",
        help="Path to the unzipped eICU demo SQLite database.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/eicu_demo_mdp",
        help="Folder where the generated MDP files will be saved.",
    )

    parser.add_argument(
        "--template-dir",
        default="data/icu_sepsis",
        help="Existing ICU-Sepsis data folder used only to mirror CSV formatting.",
    )

    parser.add_argument(
        "--num-clusters",
        type=int,
        default=120,
        help="Number of non-terminal clustered states to create before adding death/survival terminals.",
    )

    parser.add_argument(
        "--max-hours",
        type=int,
        default=48,
        help="Maximum ICU hours to use from each stay.",
    )

    parser.add_argument(
        "--min-hours-per-stay",
        type=int,
        default=3,
        help="Minimum number of hourly records required for an ICU stay to be included.",
    )

    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=4,
        help="Maximum allowed gap between consecutive hourly records when creating transitions.",
    )

    parser.add_argument(
        "--min-action-support",
        type=int,
        default=2,
        help="Minimum state-action count for an action to be marked admissible in a state.",
    )

    parser.add_argument(
        "--action-grid",
        type=int,
        choices=[5, 25],
        default=25,
        help="Number of discrete actions to build. Use 25 for the ICU-Sepsis-style two-axis action grid.",
    )

    parser.add_argument(
        "--min-admissible-actions",
        type=int,
        default=2,
        help="Minimum number of observed actions kept admissible per state when available.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Evaluation horizon stored in the metadata for compatibility with the existing project.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for state clustering.",
    )

    return parser.parse_args()


def main():
    """
    Building the eICU demo portability MDP.
    """
    args = parse_args()

    print("[INFO] Database:", args.db_path, flush=True)
    print("[INFO] Output directory:", args.output_dir, flush=True)
    print("[INFO] Template directory:", args.template_dir, flush=True)
    print("[INFO] Requested clusters:", args.num_clusters, flush=True)

    builder = EICUDemoMDPBuilder(args)
    builder.run()

# Running main() only when this file is executed directly.
if __name__ == "__main__":
    main()

