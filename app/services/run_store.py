"""Database-backed research run store.

Same public surface as the in-memory version it replaces (create_run,
get_run, update_run), so API route code didn't need to change shape --
only learn to pass a database session through.
"""

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ExecutionTrace, Finding, Report, ResearchRun
from app.schemas.research import Finding as FindingSchema


class RunStore:
    """Research run persistence backed by the research_runs/findings/reports tables."""

    async def create_run(self, session: AsyncSession, question: str) -> str:
        """Create a new run row with status "pending", returning its id."""
        run = ResearchRun(question=question, status="pending")
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return str(run.id)

    async def get_run(self, session: AsyncSession, run_id: str) -> Optional[dict[str, Any]]:
        """Return the run's data (plus report content and findings count), or None if unknown."""
        run_uuid = _parse_uuid(run_id)
        if run_uuid is None:
            return None

        run = await session.get(ResearchRun, run_uuid)
        if run is None:
            return None

        findings_count = await session.scalar(
            select(func.count()).select_from(Finding).where(Finding.run_id == run_uuid)
        )

        final_report = None
        if run.status == "completed":
            report = await session.scalar(select(Report).where(Report.run_id == run_uuid))
            final_report = report.content if report is not None else None

        return {
            "run_id": str(run.id),
            "status": run.status,
            "question": run.question,
            "created_at": run.created_at,
            "sub_questions": run.sub_questions or [],
            "research_iterations": run.research_iterations,
            "research_incomplete": run.research_incomplete,
            "final_report": final_report,
            "error": run.error,
            "findings_count": findings_count or 0,
            "total_tokens_used": run.total_tokens_used,
            "budget_exceeded": run.budget_exceeded,
        }

    async def update_run(self, session: AsyncSession, run_id: str, **fields: Any) -> None:
        """Update run columns; route "final_report" to the reports table and
        "findings" to the findings table instead of treating them as columns."""
        run_uuid = _parse_uuid(run_id)
        if run_uuid is None:
            return

        run = await session.get(ResearchRun, run_uuid)
        if run is None:
            return

        final_report = fields.pop("final_report", None)
        findings = fields.pop("findings", None)

        for key, value in fields.items():
            setattr(run, key, value)

        if final_report is not None:
            existing_report = await session.scalar(select(Report).where(Report.run_id == run_uuid))
            if existing_report is not None:
                existing_report.content = final_report
            else:
                session.add(Report(run_id=run_uuid, content=final_report))

        if findings:
            for finding in findings:
                session.add(_finding_row(run_uuid, finding))

        await session.commit()

    async def delete_run(self, session: AsyncSession, run_id: str) -> None:
        """Delete a run row, along with any findings/report rows for it.

        Used to clean up a run created by a request that then lost an
        idempotency-key race -- at "pending" status there normally aren't
        any findings/report rows yet, but the related tables are cleared
        first regardless so this stays correct if that ever changes.
        """
        run_uuid = _parse_uuid(run_id)
        if run_uuid is None:
            return

        await session.execute(delete(Finding).where(Finding.run_id == run_uuid))
        await session.execute(delete(Report).where(Report.run_id == run_uuid))
        await session.execute(delete(ExecutionTrace).where(ExecutionTrace.run_id == run_uuid))
        await session.execute(delete(ResearchRun).where(ResearchRun.id == run_uuid))
        await session.commit()


def _parse_uuid(run_id: str) -> Optional[UUID]:
    try:
        return UUID(run_id)
    except ValueError:
        return None


def _finding_row(run_uuid: UUID, finding: FindingSchema) -> Finding:
    return Finding(
        run_id=run_uuid,
        claim=finding.claim,
        source_url=finding.source_url,
        snippet=finding.snippet,
        sub_question=finding.sub_question,
        retrieved_at=finding.retrieved_at,
    )


run_store = RunStore()
