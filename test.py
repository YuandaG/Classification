import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple
from abc import ABC, abstractmethod

class DataCleaner(ABC):
    """Abstract base class for data cleaning strategies"""
    
    @abstractmethod
    def clean(self, data: np.ndarray) -> np.ndarray:
        """Clean the input data"""
        pass

class BasicDataCleaner(DataCleaner):
    """Basic data cleaning implementation"""
    
    def __init__(self, std_threshold: float = 3.0, min_value: float = None, max_value: float = None, points_per_day: int = 288):
        """
        Args:
            std_threshold: Number of standard deviations for outlier detection
            min_value: Minimum acceptable value (None for no minimum)
            max_value: Maximum acceptable value (None for no maximum)
            points_per_day: Number of points in one day
        """
        self.std_threshold = std_threshold
        self.min_value = min_value
        self.max_value = max_value
        self.points_per_day = points_per_day
    
    def clean(self, data: np.ndarray) -> np.ndarray:
        """
        Clean data by removing days containing outliers
        Args:
            data: Input time series data
        Returns:
            Cleaned data with outlier days replaced by NaN
        """
        cleaned_data = data.copy()
        
        # Reshape data into daily profiles
        n_days = len(cleaned_data) // self.points_per_day
        daily_data = cleaned_data[:n_days * self.points_per_day].reshape(n_days, self.points_per_day)
        
        # Create mask for days to exclude
        days_to_exclude = np.zeros(n_days, dtype=bool)
        
        for day in range(n_days):
            day_data = daily_data[day]
            
            # Check for statistical outliers
            mean = np.mean(day_data)
            std = np.std(day_data)
            if np.any(np.abs(day_data - mean) > (self.std_threshold * std)):
                days_to_exclude[day] = True
                continue
            
            # Check min/max bounds
            if self.min_value is not None and np.any(day_data < self.min_value):
                days_to_exclude[day] = True
                continue
            if self.max_value is not None and np.any(day_data > self.max_value):
                days_to_exclude[day] = True
                continue
        
        # Replace excluded days with NaN
        daily_data[days_to_exclude] = np.nan
        
        # Reshape back to original format
        cleaned_data[:n_days * self.points_per_day] = daily_data.reshape(-1)
        
        return cleaned_data

class TimeSeriesClassifier(ABC):
    """Abstract base class for time series classification"""
    
    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and predict clusters"""
        pass

class KMeansTimeSeriesClassifier(TimeSeriesClassifier):
    """KMeans-based time series classifier"""
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 10):
        """
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.best_n_clusters = None
        self.kmeans = None
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering result"""
        from sklearn.metrics import silhouette_score
        try:
            return silhouette_score(data, labels)
        except:
            return -1  # Return -1 if calculation fails
    
    def fit_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit KMeans and predict clusters, automatically selecting optimal number of clusters
        Args:
            data: Daily profiles array of shape (n_days, points_per_day)
        Returns:
            Tuple of (cluster labels, cluster centers)
        """
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Try different numbers of clusters
        best_score = -1
        best_labels = None
        best_centers = None
        
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            # Skip if we have fewer samples than clusters
            if len(data) < n_clusters:
                continue
                
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data_scaled)
            
            # Calculate silhouette score
            score = self._calculate_silhouette_score(data_scaled, labels)
            
            # Update best result if score is better
            if score > best_score:
                best_score = score
                best_labels = labels
                self.best_n_clusters = n_clusters
                self.kmeans = kmeans
        
        if best_labels is None:
            # If no valid clustering found, use minimum number of clusters
            self.best_n_clusters = self.min_clusters
            self.kmeans = KMeans(n_clusters=self.min_clusters, random_state=42)
            best_labels = self.kmeans.fit_predict(data_scaled)
        
        # Get cluster centers and inverse transform to original scale
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        print(f"Selected {self.best_n_clusters} clusters with silhouette score: {best_score:.3f}")
        return best_labels, centers

class TimeSeriesAnalyzer:
    """Main class for time series analysis"""
    
    def __init__(self, 
                 points_per_day: int,
                 cleaner: DataCleaner = None,
                 classifier: TimeSeriesClassifier = None):
        """
        Args:
            points_per_day: Number of points in one day
            cleaner: Data cleaning strategy (optional)
            classifier: Classification strategy (optional)
        """
        self.points_per_day = points_per_day
        self.cleaner = cleaner or BasicDataCleaner()
        self.classifier = classifier or KMeansTimeSeriesClassifier()
    
    def reshape_to_daily_profiles(self, data: np.ndarray) -> np.ndarray:
        """Reshape data into daily profiles"""
        n_days = len(data) // self.points_per_day
        if n_days == 0:
            raise ValueError(f"Input data length ({len(data)}) is shorter than points_per_day ({self.points_per_day})")
        return data[:n_days * self.points_per_day].reshape(n_days, self.points_per_day)
    
    def normalize_daily_profiles(self, daily_profiles: np.ndarray) -> np.ndarray:
        """
        Normalize each daily profile to have values between 0 and 1
        Args:
            daily_profiles: Array of shape (n_days, points_per_day)
        Returns:
            Normalized daily profiles
        """
        normalized = np.zeros_like(daily_profiles, dtype=float)
        for i in range(len(daily_profiles)):
            min_val = np.min(daily_profiles[i])
            max_val = np.max(daily_profiles[i])
            if max_val > min_val:  # Avoid division by zero
                normalized[i] = (daily_profiles[i] - min_val) / (max_val - min_val)
            else:
                normalized[i] = daily_profiles[i] - min_val  # If all values are the same
        return normalized
    
    def analyze(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze time series data
        Args:
            data: Input time series data
        Returns:
            Tuple of (cleaned_data, cluster_labels, cluster_centers)
        """
        # Clean data
        cleaned_data = self.cleaner.clean(data)
        
        # Reshape into daily profiles
        daily_profiles = self.reshape_to_daily_profiles(cleaned_data)
        
        # Remove days containing NaN values
        valid_days_mask = ~np.any(np.isnan(daily_profiles), axis=1)
        valid_daily_profiles = daily_profiles[valid_days_mask]
        
        if len(valid_daily_profiles) == 0:
            raise ValueError("No valid days remaining after cleaning. Consider adjusting cleaning parameters.")
        
        # Normalize valid daily profiles
        normalized_profiles = self.normalize_daily_profiles(valid_daily_profiles)
        
        # Classify using normalized profiles
        labels, centers = self.classifier.fit_predict(normalized_profiles)
        
        # Create full labels array (including NaN days)
        full_labels = np.full(len(daily_profiles), -1)  # -1 for invalid days
        full_labels[valid_days_mask] = labels
        
        # Denormalize centers to match original scale
        denormalized_centers = np.zeros_like(centers)
        for i in range(len(centers)):
            cluster_profiles = valid_daily_profiles[labels == i]
            if len(cluster_profiles) > 0:
                # Scale the center pattern using the average min/max of the cluster
                avg_min = np.mean([np.min(profile) for profile in cluster_profiles])
                avg_max = np.mean([np.max(profile) for profile in cluster_profiles])
                denormalized_centers[i] = centers[i] * (avg_max - avg_min) + avg_min
        
        return cleaned_data, full_labels, denormalized_centers
    
    def plot_results(self, data: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        """Plot classification results for both original and normalized data"""
        daily_profiles = self.reshape_to_daily_profiles(data)
        
        # Only use valid days for plotting
        valid_days_mask = ~np.any(np.isnan(daily_profiles), axis=1)
        valid_daily_profiles = daily_profiles[valid_days_mask]
        valid_labels = labels[valid_days_mask]
        
        normalized_profiles = self.normalize_daily_profiles(valid_daily_profiles)
        n_clusters = len(centers)
        
        # Create two sets of subplots: one for original data, one for normalized
        fig1, axes1 = plt.subplots(n_clusters, 1, figsize=(12, 4*n_clusters))
        fig2, axes2 = plt.subplots(n_clusters, 1, figsize=(12, 4*n_clusters))
        
        if n_clusters == 1:
            axes1, axes2 = [axes1], [axes2]
        
        x_ticks = np.linspace(0, self.points_per_day-1, 6)
        x_labels = [f'{int(x/self.points_per_day*24)}:00' for x in x_ticks]
        
        # Plot original data
        for i in range(n_clusters):
            ax = axes1[i]
            cluster_profiles = valid_daily_profiles[valid_labels == i]
            for profile in cluster_profiles:
                ax.plot(profile, 'gray', alpha=0.3)
            ax.plot(centers[i], 'r-', linewidth=2, label=f'Cluster {i+1} Center')
            ax.set_title(f'Cluster {i+1} (n={len(cluster_profiles)} days) - Original Scale')
            ax.grid(True)
            ax.legend()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('Value')
        
        # Plot normalized data
        normalized_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            ax = axes2[i]
            cluster_profiles = normalized_profiles[valid_labels == i]
            for profile in cluster_profiles:
                ax.plot(profile, 'gray', alpha=0.3)
            
            # Normalize the center for this plot
            cluster_center = centers[i]
            min_val = np.min(cluster_center)
            max_val = np.max(cluster_center)
            if max_val > min_val:
                normalized_centers[i] = (cluster_center - min_val) / (max_val - min_val)
            else:
                normalized_centers[i] = cluster_center - min_val
            
            ax.plot(normalized_centers[i], 'r-', linewidth=2, label=f'Cluster {i+1} Center')
            ax.set_title(f'Cluster {i+1} (n={len(cluster_profiles)} days) - Normalized')
            ax.grid(True)
            ax.legend()
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('Normalized Value')
            ax.set_ylim(-0.1, 1.1)  # Add some padding to the normalized plot
        
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.savefig('classification_results_original.png')
        fig2.savefig('classification_results_normalized.png')
        plt.close('all')


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('INPUT/data_220.csv')
    data = df.values.flatten()
    
    # Create analyzer with parameters tuned for data_220.csv
    analyzer = TimeSeriesAnalyzer(
        points_per_day=288,  # 48 points per day (30-minute intervals)
        cleaner=BasicDataCleaner(
            std_threshold=4.0,  # Remove outliers beyond 4 standard deviations
            min_value=10,       # Data shouldn't be negative
            max_value=90000,    # No upper limit
            points_per_day=288  # Add points_per_day parameter
        ),
        classifier=KMeansTimeSeriesClassifier(min_clusters=2, max_clusters=10)
    )
    
    # Print original data summary
    n_days = len(data) // analyzer.points_per_day
    remaining_points = len(data) % analyzer.points_per_day
    print(f"\nOriginal Data Summary:")
    print(f"Total points: {len(data)}")
    print(f"Complete days: {n_days}")
    print(f"Remaining points (dropped): {remaining_points}")
    print(f"Data range: {data.min():.2f} to {data.max():.2f}")
    
    # Analyze data
    cleaned_data, labels, centers = analyzer.analyze(data)
    
    # Print cleaning summary
    n_modified = np.sum(data != cleaned_data)
    n_invalid_days = np.sum(labels == -1)  # Count days marked as invalid
    print(f"\nCleaning Summary:")
    print(f"Points modified: {n_modified} ({n_modified/len(data)*100:.2f}%)")
    print(f"Invalid days removed: {n_invalid_days}")
    print(f"Cleaned data range: {cleaned_data.min():.2f} to {cleaned_data.max():.2f}")
    
    # Plot original vs cleaned data for first few days
    plt.figure(figsize=(15, 6))
    days_to_show = min(3, n_days)
    points_to_show = days_to_show * analyzer.points_per_day
    
    plt.plot(data[:points_to_show], 'b-', alpha=0.5, label='Original')
    plt.plot(cleaned_data[:points_to_show], 'r-', label='Cleaned')
    
    # Add day markers
    for i in range(days_to_show):
        plt.axvline(x=i*analyzer.points_per_day, color='gray', linestyle='--', alpha=0.5)
        plt.text(i*analyzer.points_per_day, plt.ylim()[1], f'Day {i+1}', 
                rotation=0, ha='right', va='bottom')
    
    plt.title('Original vs Cleaned Data (First 3 Days)')
    plt.grid(True)
    plt.legend()
    plt.savefig('data_cleaning_comparison.png')
    plt.close()
    
    # Plot classification results
    analyzer.plot_results(cleaned_data, labels, centers)
    
    # Print detailed classification summary
    valid_labels = labels[labels != -1]  # Only consider valid days
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    print("\nClassification Summary:")
    print(f"Total valid days: {len(valid_labels)}")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label + 1}: {count} days ({count/len(valid_labels)*100:.1f}%)")
    
    # Save results to CSV with invalid days marked
    results_df = pd.DataFrame({
        'day': range(1, len(labels) + 1),
        'cluster': np.where(labels == -1, 'Invalid', labels + 1),  # Mark invalid days
    })
    results_df.to_csv('classification_results.csv', index=False)
