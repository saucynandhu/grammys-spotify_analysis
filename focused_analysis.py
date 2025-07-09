import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict

# Set up the plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Create output directory for plots
os.makedirs('analysis_plots', exist_ok=True)

def clean_artist_name(artist):
    """Clean artist names by removing featured artists and extra information."""
    if pd.isna(artist):
        return ""
    # Remove content in parentheses and brackets
    artist = re.sub(r'\([^)]*\)', '', str(artist))
    artist = re.sub(r'\[[^]]*\]', '', artist)
    # Remove common suffixes
    artist = re.sub(r'\s+feat\..*$', '', artist, flags=re.IGNORECASE)
    artist = re.sub(r'\s+&.*$', '', artist, flags=re.IGNORECASE)
    artist = re.sub(r'\s+x\s+.*$', '', artist, flags=re.IGNORECASE)
    # Clean up whitespace
    artist = artist.strip()
    return artist

def load_and_preprocess_data():
    """Load and preprocess all datasets."""
    print("Loading and preprocessing data...")
    
    # Load datasets
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    
    # Load Grammy data
    grammy = pd.read_csv(os.path.join(datasets_dir, 'Grammy Award Nominees and Winners 1958-2024.csv'), 
                        encoding='latin1')
    
    # Load Spotify most streamed songs
    spotify_songs = pd.read_csv(os.path.join(datasets_dir, 'Spotify most streamed.csv'), 
                              thousands=',',
                              encoding='latin1')
    
    # Clean data
    grammy['clean_nominee'] = grammy['Nominee'].apply(clean_artist_name)
    
    # Extract artist and song from 'Artist and Title' column
    spotify_songs[['artist', 'song']] = spotify_songs['Artist and Title'].str.split(' - ', n=1, expand=True)
    spotify_songs['clean_artist'] = spotify_songs['artist'].apply(clean_artist_name)
    
    # Convert streams to millions for easier interpretation
    spotify_songs['streams_millions'] = spotify_songs['Streams'] / 1000000
    
    return grammy, spotify_songs

def analyze_grammy_impact(grammy, spotify_songs):
    """Analyze the relationship between Grammy wins and streaming success."""
    print("\n=== Grammy Impact on Streaming Success ===")
    
    # Get unique Grammy winners
    winners = grammy[grammy['Winner'] == True]['clean_nominee'].unique()
    
    # Add a column to indicate if artist has won a Grammy
    spotify_songs['has_grammy'] = spotify_songs['clean_artist'].isin(winners)
    
    # Calculate average streams by Grammy status
    avg_streams = spotify_songs.groupby('has_grammy')['streams_millions'].mean()
    
    print(f"Average streams (millions):")
    print(f"Grammy winners: {avg_streams[True]:.2f}")
    print(f"Non-winners: {avg_streams[False]:.2f}")
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_streams.index.map({True: 'Grammy Winner', False: 'Non-Winner'}), 
                y=avg_streams.values)
    plt.title('Average Streaming Numbers by Grammy Status')
    plt.ylabel('Average Streams (Millions)')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('analysis_plots/grammy_impact.png')
    plt.close()
    
    # Find top streamed artists without Grammys
    top_non_grammy = spotify_songs[~spotify_songs['has_grammy']].nlargest(10, 'streams_millions')
    
    print("\nTop 10 Streamed Artists Without Grammy Wins:")
    print(top_non_grammy[['artist', 'song', 'streams_millions']].to_string(index=False))
    
    # Save to CSV for reference
    top_non_grammy.to_csv('analysis_plots/top_non_grammy_artists.csv', index=False)

def analyze_genre_trends(grammy):
    """Analyze genre trends in Grammy awards."""
    print("\n=== Grammy Genre Trends ===")
    
    # Extract genre from award names (simplified)
    grammy['award_genre'] = grammy['Award Name'].str.extract(
        r'(Pop|Rock|Rap|R&B|Country|Jazz|Classical|Dance|Latin|Alternative|Metal|Gospel|Reggae)', 
        flags=re.IGNORECASE, expand=False
    )
    
    # Count awards by genre
    genre_counts = grammy['award_genre'].value_counts().head(10)
    
    # Plot top genres
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar')
    plt.title('Top 10 Most Common Grammy Award Genres')
    plt.xlabel('Genre')
    plt.ylabel('Number of Awards')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('analysis_plots/grammy_genres.png')
    plt.close()
    
    print("\nTop Grammy award genres:")
    print(genre_counts)

def analyze_artist_longevity(grammy, spotify_songs):
    """Analyze artist longevity in terms of Grammy recognition and streaming."""
    print("\n=== Artist Longevity Analysis ===")
    
    # Get artists with Grammy wins by year
    grammy_winners = grammy[grammy['Winner'] == True][['Year', 'clean_nominee']].drop_duplicates()
    
    # Count years active (years with Grammy wins)
    years_active = grammy_winners.groupby('clean_nominee')['Year'].agg(['min', 'max', 'count'])
    years_active['career_span'] = years_active['max'] - years_active['min'] + 1
    
    # Get top artists by career span
    top_longevity = years_active.sort_values('career_span', ascending=False).head(10)
    
    print("\nArtists with Longest Grammy Recognition Spans:")
    print(top_longevity[['min', 'max', 'career_span', 'count']]\
          .rename(columns={'min': 'First Win', 'max': 'Last Win', 
                          'career_span': 'Years Active', 'count': 'Total Wins'}))
    
    # Plot career spans
    plt.figure(figsize=(12, 6))
    top_longevity = top_longevity.sort_values('career_span')
    plt.barh(top_longevity.index, top_longevity['career_span'])
    plt.title('Top 10 Artists by Grammy Recognition Span (Years)')
    plt.xlabel('Years of Grammy Recognition')
    plt.tight_layout()
    plt.savefig('analysis_plots/artist_longevity.png')
    plt.close()

def main():
    # Load and preprocess data
    grammy, spotify_songs = load_and_preprocess_data()
    
    # Perform analyses
    analyze_grammy_impact(grammy, spotify_songs)
    analyze_genre_trends(grammy)
    analyze_artist_longevity(grammy, spotify_songs)
    
    print("\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")

if __name__ == "__main__":
    main()
