#[path = "../../build-support/models.rs"]
mod models;

use models::OnnxModel;

const MODELS: [OnnxModel; 17] = [
    OnnxModel {
        stem: "all-MiniLM-L6-v2",
        cfg: "has_minilm",
        weights_env: "MINILM_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-text",
        cfg: "has_jina_text",
        weights_env: "JINA_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "jina-clip-v1-vision",
        cfg: "has_jina_vision",
        weights_env: "JINA_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-text",
        cfg: "has_bioclip_text",
        weights_env: "BIOCLIP_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "bioclip-vision",
        cfg: "has_bioclip_vision",
        weights_env: "BIOCLIP_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "nomic-embed-text-v1_5",
        cfg: "has_nomic_text",
        weights_env: "NOMIC_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "nomic-embed-vision-v1_5",
        cfg: "has_nomic_vision",
        weights_env: "NOMIC_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-t16-384-text",
        cfg: "has_pe_core_t_text",
        weights_env: "PE_CORE_T_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-t16-384-vision",
        cfg: "has_pe_core_t_vision",
        weights_env: "PE_CORE_T_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-s16-384-text",
        cfg: "has_pe_core_s_text",
        weights_env: "PE_CORE_S_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-s16-384-vision",
        cfg: "has_pe_core_s_vision",
        weights_env: "PE_CORE_S_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-b16-224-text",
        cfg: "has_pe_core_b_text",
        weights_env: "PE_CORE_B_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-b16-224-vision",
        cfg: "has_pe_core_b_vision",
        weights_env: "PE_CORE_B_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-l14-336-text",
        cfg: "has_pe_core_l_text",
        weights_env: "PE_CORE_L_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-l14-336-vision",
        cfg: "has_pe_core_l_vision",
        weights_env: "PE_CORE_L_VISION_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-g14-448-text",
        cfg: "has_pe_core_g_text",
        weights_env: "PE_CORE_G_TEXT_WEIGHTS",
    },
    OnnxModel {
        stem: "pe-core-g14-448-vision",
        cfg: "has_pe_core_g_vision",
        weights_env: "PE_CORE_G_VISION_WEIGHTS",
    },
];

fn main() {
    for model in &MODELS {
        models::import_if_present(model);
    }
}
