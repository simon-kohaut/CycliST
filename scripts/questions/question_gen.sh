#!/bin/bash
# filepath: /workspaces/CycliST/run_question_debug.sh

universal_templates=("universal_descriptive_attributes" "universal_descriptive_compare" "universal_descriptive_relate")
existential_templates=("existential_descriptive_attributes" "existential_descriptive_compare" "existential_descriptive_relate")
cyclic_representative=("cycle_representative_attribute" "cycle_representative_orbit")
counting=("counting_frequency" "counting_occurance")

template_selection=("${universal_templates[@]}" "${existential_templates[@]}" "${cyclic_representative[@]}" "${counting[@]}")

for split in "train" "test" "val"
do
    for dataset in "unicycle" # "bicycle" "tricycle" "unicycle_cluttered"
    do 
        for template in "${template_selection[@]}"
        do
            echo "Generating questions for template: $template, dataset: $dataset, split: $split"
            # Run the Python script with the specified arguments
            python3 /workspaces/CycliST_dev/CycliST/cyclist/questions/generate_questions.py \
                --synonyms_json "/workspaces/CycliST_dev/CycliST/assets/question_templates/synonyms.json" \
                --cyclic_json "/workspaces/CycliST_dev/CycliST/assets/question_templates/cyclic.json" \
                --metadata_file "/workspaces/CycliST_dev/CycliST/assets/question_templates/metadata.json" \
                --template_dir "/workspaces/CycliST_dev/CycliST/assets/question_templates/CycliST_1.0_templates" \
                --input_scene_file "/workspaces/CycliST_dev/CycliST/output/scenes/${dataset}/${split}" \
                --output_questions_file "/workspaces/CycliST_dev/CycliST/output/questions/${dataset}/${split}/questions_${template}.json" \
                --templates_per_image "1" \
                --templates "$template"
        done
    done
done