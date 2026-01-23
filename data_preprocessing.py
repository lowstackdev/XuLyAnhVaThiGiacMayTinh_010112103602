import pandas as pd
import ast

# Priority D, G, C, A, H, M > O (Other) > N (Normal)
EYE_DISEASES = {
    'A': ['Age-related Macular Degeneration'],
    'B': [],
    'C': ['Cataract'],
    'D': ['Diabetes'],
    'E': [],
    'F': [],
    'G': ['Glaucoma'],
    'H': ['Hypertension'],
    'I': [],
    'J': [],
    'K': [],
    'L': [],
    'M': ['Pathological Myopia'],
    'N': ['Normal'],
    'O': ['Other diseases/abnormalities'],
    'P': [],
    'Q': [],
    'R': [],
    'S': [],
    'T': [],
    'U': [],
    'V': [],
    'W': [],
    'X': [],
    'Y': [],
    'Z': []
}

def balance_dataset(df, label_column='label'):
    df = df.copy()

    df['__parsed_label__'] = df[label_column].apply(ast.literal_eval)

    df['__main_label__'] = df['__parsed_label__'].apply(lambda x: x[0] if x else None)

    df = df.dropna(subset=['__main_label__'])

    df['__is_complete__'] = df.isnull().sum(axis=1) == 0

    grouped = df.groupby('__main_label__')

    min_count = grouped.size().min()

    balanced = []
    for name, group in grouped:
        complete = group[group['__is_complete__']]
        incomplete = group[~group['__is_complete__']]

        num_complete = min(len(complete), min_count)
        sampled = complete.sample(n=num_complete, random_state=42)

        remaining = min_count - num_complete
        if remaining > 0:
            sampled = pd.concat([sampled, incomplete.sample(n=remaining, random_state=42)])

        balanced.append(sampled)

    balanced_df = pd.concat(balanced).reset_index(drop=True)

    balanced_df = balanced_df.drop(columns=['__parsed_label__', '__main_label__', '__is_complete__'])

    return balanced_df
