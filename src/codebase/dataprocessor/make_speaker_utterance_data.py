import re

import polars as pl


def make_assembly_data(minutes_json):
    id_num = 0
    label_num = 1
    df_full = pl.DataFrame(
        {"id": [], "speaker_name": [], "utterance": [], "label": []},
        {
            "id": pl.Int64,
            "speaker_name": pl.Utf8,
            "utterance": pl.Utf8,
            "label": pl.Int64,
        },
    )

    for i, utterances in enumerate(minutes_json["assembly"]["utterances"]):
        # if i > 5:
        #     break
        # speaker_name = utterances["speaker"]
        if re.compile(r"◯.+番（.+）").search(utterances["speaker"]):
            speaker_name = re.sub(
                "（|）|君",
                "",
                re.search(r"（\D+君）", utterances["speaker"]).group(),
            )
        else:
            speaker_name = re.sub(
                r"◯|（.+）",
                "",
                utterances["speaker"]
                # re.search(r"（\D+君）", utterances["speaker"]).group(),
            )

        for utterance in utterances["utterance"]:
            utterance_split = re.findall("[^。！？!?、]+[。！？!?、]?", utterance)
            utterance_split_list = [a for a in utterance_split if a != ""]
            for utterance_split in utterance_split_list:
                df = pl.DataFrame(
                    {
                        "id": id_num,
                        "speaker_name": speaker_name,
                        "utterance": utterance_split,
                        "label": label_num,
                    }
                )
                df_full = pl.concat([df_full, df])
                id_num += 1
            label_num += 1

    return df_full


def make_digest_data(minutes_json):
    id_num = 0
    df_full = pl.DataFrame(
        {"id": [], "speaker_name": [], "utterance": [], "label": []},
        {
            "id": pl.Int64,
            "speaker_name": pl.Utf8,
            "utterance": pl.Utf8,
            "label": pl.Utf8,
        },
    )

    for i, speakers in enumerate(minutes_json["digest"]["speakers"]):
        # if i > 5:
        #     break
        # speaker_name = speakers["speaker_name"]
        speaker_name = re.sub(r"（.+）", "", speakers["speaker_name"])
        summury = speakers["summury"]
        label = "speaker.summury"
        df = pl.DataFrame(
            {
                "id": id_num,
                "speaker_name": speaker_name,
                "utterance": summury,
                "label": label,
            }
        )
        df_full = pl.concat([df_full, df])
        id_num += 1
        for policies in speakers["policies"]:
            for key, value in policies.items():
                if key == "title":
                    df = pl.DataFrame(
                        {
                            "id": id_num,
                            "speaker_name": speaker_name,
                            "utterance": value,
                            "label": f"policy.{key}",
                        }
                    )
                    df_full = pl.concat([df_full, df])
                    id_num += 1
                elif key == "utterance":
                    utterances = [
                        a for a in re.split(r"[(〔]\d+[〕)]", value) if a != ""
                    ]
                    for utterance in utterances:
                        df = pl.DataFrame(
                            {
                                "id": id_num,
                                "speaker_name": speaker_name,
                                "utterance": utterance,
                                "label": f"policy.{key}",
                            }
                        )
                        df_full = pl.concat([df_full, df])
                        id_num += 1
                elif key == "replies":
                    for reply in value:
                        reply_value = list(reply.values())
                        utterances = [
                            a
                            for a in re.split(r"[(〔]\d+[〕)]", reply_value[1])
                            if a != ""
                        ]
                        for utterance in utterances:
                            df = pl.DataFrame(
                                {
                                    "id": id_num,
                                    "speaker_name": reply_value[0],
                                    "utterance": utterance,
                                    "label": "reply.utterance",
                                }
                            )
                            df_full = pl.concat([df_full, df])
                            id_num += 1

    return df_full
