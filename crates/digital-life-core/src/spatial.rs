use crate::agent::Agent;
use rstar::{RTree, RTreeObject, AABB};

/// Lightweight position-only struct for spatial indexing to avoid cloning full agents.
#[derive(Clone, Debug)]
pub struct AgentLocation {
    pub id: u32,
    pub position: [f64; 2],
}

impl RTreeObject for AgentLocation {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.position)
    }
}

/// Build an R*-tree from agent positions via bulk_load (O(n log n)).
pub fn build_index(agents: &[Agent]) -> RTree<AgentLocation> {
    let locations: Vec<AgentLocation> = agents
        .iter()
        .map(|a| AgentLocation {
            id: a.id,
            position: a.position,
        })
        .collect();
    RTree::bulk_load(locations)
}

/// Count neighbors within `radius` of `center` (excludes agent with `self_id`).
/// Avoids allocation — returns count only.
pub fn count_neighbors(
    tree: &RTree<AgentLocation>,
    center: [f64; 2],
    radius: f64,
    self_id: u32,
) -> usize {
    let envelope = AABB::from_corners(
        [center[0] - radius, center[1] - radius],
        [center[0] + radius, center[1] + radius],
    );
    let r_sq = radius * radius;

    tree.locate_in_envelope(&envelope)
        .filter(|loc| {
            if loc.id == self_id {
                return false;
            }
            let dx = loc.position[0] - center[0];
            let dy = loc.position[1] - center[1];
            dx * dx + dy * dy <= r_sq
        })
        .count()
}

/// Query neighbors within `radius` of `center`, returning their agent IDs.
/// Uses AABB envelope query then filters by Euclidean distance.
/// Excludes the agent with `self_id`.
pub fn query_neighbors(
    tree: &RTree<AgentLocation>,
    center: [f64; 2],
    radius: f64,
    self_id: u32,
) -> Vec<u32> {
    let envelope = AABB::from_corners(
        [center[0] - radius, center[1] - radius],
        [center[0] + radius, center[1] + radius],
    );
    let r_sq = radius * radius;

    tree.locate_in_envelope(&envelope)
        .filter(|loc| {
            if loc.id == self_id {
                return false;
            }
            let dx = loc.position[0] - center[0];
            let dy = loc.position[1] - center[1];
            dx * dx + dy * dy <= r_sq
        })
        .map(|loc| loc.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agent(id: u32, x: f64, y: f64) -> Agent {
        Agent::new(id, 0, [x, y])
    }

    #[test]
    fn query_finds_agents_within_radius() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),  // distance 1.0
            make_agent(2, 50.0, 50.0), // far away
        ];
        let tree = build_index(&agents);
        let mut result = query_neighbors(&tree, [5.0, 5.0], 2.0, u32::MAX);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn query_excludes_self() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),
        ];
        let tree = build_index(&agents);
        let result = query_neighbors(&tree, [5.0, 5.0], 2.0, 0);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn query_excludes_agents_outside_radius() {
        let agents = vec![
            make_agent(0, 0.0, 0.0),
            make_agent(1, 10.0, 10.0),
        ];
        let tree = build_index(&agents);
        let result = query_neighbors(&tree, [0.0, 0.0], 1.0, u32::MAX);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn query_returns_agent_ids_not_indices() {
        let agents = vec![
            make_agent(42, 1.0, 1.0),
            make_agent(99, 1.5, 1.0),
        ];
        let tree = build_index(&agents);
        let mut result = query_neighbors(&tree, [1.0, 1.0], 2.0, u32::MAX);
        result.sort();
        assert_eq!(result, vec![42, 99]);
    }

    #[test]
    fn count_neighbors_excludes_self() {
        let agents = vec![
            make_agent(0, 5.0, 5.0),
            make_agent(1, 6.0, 5.0),
            make_agent(2, 50.0, 50.0),
        ];
        let tree = build_index(&agents);
        assert_eq!(count_neighbors(&tree, [5.0, 5.0], 2.0, 0), 1);
    }
}
