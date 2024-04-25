
def load_wine_dataset():
    wine_data = load_wine()
    df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    return df


def handle_missing_values(df):
    # Check for missing values
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
    else:
        # Handle missing values
        df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values
        print("Missing values handled.")


def generate_data_profile(df):
    profile = ProfileReport(df, title="Wine Dataset Profiling Report", explorative=True)
    profile.to_file("wine_dataset_profile.html")
    print("Data profiling report generated.")


def preprocess_data(df):
    # Handling missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)  # Example: Forward fill missing values

    # Feature scaling or normalization
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    # Splitting dataset into training and testing sets
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test