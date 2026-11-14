"""Version official glove records by atomic directory moves, without deletion."""
import json
import uuid
from pathlib import Path


def _pointer(home,archive):
    target=home/'previous-official-calibration.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps({'archive':str(archive.relative_to(home))}))
    temp.replace(target)


def archive_calibration(home):
    home=Path(home)
    if not (home/'glove').is_dir():raise ValueError('没有可备份的官方记录，请继续首次八动作采集')
    archive=home/'calibration-history'/uuid.uuid4().hex
    archive.mkdir(parents=True)
    (home/'glove').rename(archive/'glove')
    _pointer(home,archive)
    return archive


def restore_calibration(home):
    home=Path(home)
    pointer=home/'previous-official-calibration.json'
    if not pointer.is_file():raise ValueError('没有上一份官方标定记录')
    archive=(home/json.loads(pointer.read_text())['archive']).resolve()
    history=(home/'calibration-history').resolve()
    if archive.parent!=history or not (archive/'glove').is_dir():raise ValueError('上一份官方标定记录不可用')
    # Preserve the new capture too, including a partially completed attempt.
    current=archive_calibration(home) if (home/'glove').exists() else None
    (archive/'glove').rename(home/'glove')
    if current is None:pointer.unlink()
    return archive
