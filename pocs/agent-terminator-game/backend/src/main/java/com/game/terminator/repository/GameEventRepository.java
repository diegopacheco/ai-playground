package com.game.terminator.repository;

import com.game.terminator.model.GameEvent;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface GameEventRepository extends CrudRepository<GameEvent, Long> {
    List<GameEvent> findByGameIdOrderByCycleAsc(String gameId);
}
