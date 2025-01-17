import React, { useState, useEffect } from 'react';
import { getIncidents } from '../service/api';
import { Loading, Button } from '@/components';
import styles from './knowledge.module.less';

export function IncidentData() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({
    skip: 0,
    limit: 10,
    total: 0,
    hasMore: false
  });

  async function fetchIncidents(isLoadMore = false) {
    try {
      setLoading(true);
      const response = await getIncidents(pagination.skip, pagination.limit);
      
      setIncidents(prev => isLoadMore ? [...prev, ...response.documents] : response.documents);
      setPagination({
        skip: response.skip,
        limit: response.limit,
        total: response.total,
        hasMore: response.has_more
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchIncidents(false);
  }, []); // Initial load

  const loadMore = async () => {
    setPagination(prev => ({
      ...prev,
      skip: prev.skip + prev.limit
    }));
    await fetchIncidents(true);
  };

  if (error) return <div className={styles.error}>Error loading incidents: {error}</div>;

  return (
    <div className={styles.incidents}>
      <h2 className={styles.incidentsTitle}>Incident Data</h2>
      <div className={styles.incidentsList}>
        {incidents.map((incident, index) => (
          <div key={`${incident.metadata?.filename}-${index}`} className={styles.incidentItem}>
            <div className={styles.incidentContent}>
              {incident.metadata?.preview_image && (
                <div className={styles.incidentFilePreview}>
                  <img 
                    src={`data:image/png;base64,${incident.metadata.preview_image}`}
                    alt={`Preview of ${incident.metadata.filename}`}
                  />
                </div>
              )}
            </div>
            <div className={styles.fileInfo}>
              <div className={styles.fileName}>
                {incident.metadata?.filename}
              </div>
              <div className={styles.documentCount}>
                {incident.metadata?.embedding_count} embeddings
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {loading && <Loading />}
      
      {pagination.hasMore && !loading && (
        <div className={styles.loadMore}>
          <Button onClick={loadMore} type="primary">Load More</Button>
        </div>
      )}
    </div>
  );
}