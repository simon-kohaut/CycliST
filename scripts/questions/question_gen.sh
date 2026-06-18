#!/bin/bash

base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

universal_templates=("universal_descriptive_attributes" "universal_descriptive_compare" "universal_descriptive_relate")
existential_templates=("existential_descriptive_attributes" "existential_descriptive_compare" "existential_descriptive_relate")
cyclic_representative=("cycle_representative_orbit" "cycle_representative_clockwise" "cycle_representative_transition")
counting=("counting_cycles" "counting_frequency" "counting_occurence")

template_selection=("${universal_templates[@]}" "${existential_templates[@]}" "${cyclic_representative[@]}" "${counting[@]}")
# Single-template override for debugging:
# template_selection=("cycle_representative_clockwise")

for split in "train" "test" "val"
do
    for dataset in "unicycle" "bicycle" "tricycle" "unicycle_cluttered" "nightrider"
    do
        for template in "${template_selection[@]}"
        do
            echo "Generating questions for template: $template, dataset: $dataset, split: $split"
            mkdir -p "$base_path/output/questions/${dataset}/${split}"
            python3 "$base_path/cyclist/questions/generate_questions.py" \
                --synonyms_json "$base_path/assets/question_templates/synonyms.json" \
                --cyclic_json "$base_path/assets/question_templates/cyclic.json" \
                --metadata_file "$base_path/assets/question_templates/metadata.json" \
                --template_dir "$base_path/assets/question_templates/CycliST_1.0_templates" \
                --input_scene_file "$base_path/output/scenes/${dataset}/${split}" \
                --output_questions_file "$base_path/output/questions/${dataset}/${split}/questions_${template}.json" \
                --templates_per_image "1" \
                --templates "$template"
        done
    done
done
